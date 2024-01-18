
using Microsoft.SemanticKernel.Memory;
using Microsoft.SemanticKernel.Plugins.Memory;
using LLamaSharp.SemanticKernel.TextCompletion;
using LLamaSharp.SemanticKernel.TextEmbedding;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using LLama.Common;
using Microsoft.SemanticKernel.TextGeneration;
using LLamaSharp.SemanticKernel.ChatCompletion;

namespace LLama.Examples.Examples
{
    public class SemanticKernelFileChat
    {
        public const string DocumentsPath = "./Assets/lluhay-war-story.txt";
        public const string MemoryCollectionName = "StoryDocuments";

        public static async Task Run()
        {
            Console.Write("Please input your model path: ");
            var modelPath = Console.ReadLine();

            var seed = 1337u;
            var parameters = new ModelParams(modelPath)
            {
                Seed = seed,
                EmbeddingMode = true
            };

            using var model = LLamaWeights.LoadFromFile(parameters);
            var embedder = new LLamaEmbedder(model, parameters);
            var ex = new StatelessExecutor(model, parameters);

            var builder = Kernel.CreateBuilder();
            builder.Services.AddKeyedSingleton<ITextGenerationService>("local-llama", new LLamaSharpTextCompletion(ex));

            var memory = new MemoryBuilder()
                .WithTextEmbeddingGeneration(new LLamaSharpEmbeddingGeneration(embedder))
                .WithMemoryStore(new VolatileMemoryStore())
                .Build();

            var kernel = builder.Build();

            var fileText = File.ReadAllText(DocumentsPath);
            var fileName = Path.GetFileName(DocumentsPath);

            await memory.SaveInformationAsync(
                collection: MemoryCollectionName,
                id: fileName,
                text: fileText);

            // START Retrive memories
            var lookup = await memory.GetAsync(MemoryCollectionName, fileName);
            Console.WriteLine("Memory with key 'info1':" + lookup?.Metadata.Text ?? "ERROR: memory not found");
            Console.WriteLine();
            // END Retrive

            // START Retrive TextMemoryPlugin
            var memoryPlugin = kernel.ImportPluginFromObject(new TextMemoryPlugin(memory));

            var memoryRetrive = await kernel.InvokeAsync(memoryPlugin["Retrieve"], new KernelArguments()
            {
                [TextMemoryPlugin.CollectionParam] = MemoryCollectionName,
                [TextMemoryPlugin.KeyParam] = fileName,
            });

            Console.WriteLine("Retrive MemoryPlugin with key 'info5':" + memoryRetrive.GetValue<string>() ??
                              "ERROR: memory not found");
            Console.WriteLine();
            // END TextMemoryPlugin

            // START Recall from memory
            Console.WriteLine("== PART 3: Recall (similarity search) with AI Embeddings ==");
            await foreach (var answer in memory.SearchAsync(
                               collection: MemoryCollectionName,
                               query: "lluhay-war-story.txt",
                               limit: 2,
                               minRelevanceScore: 0.09,
                               withEmbeddings: true,
                               cancellationToken: CancellationToken.None))
            {
                Console.WriteLine($"Answer: {answer.Metadata.Text}");
            }
            // END Recall

            // Recall document input
            var prompt = """
                         Consider only details below when answering questions

                         BEGIN DETAILS
                         {{recall 'lluhay-war-story.txt'}}
                         END DETAILS

                         Question: {{$input}}

                         Answer:
                         """;

            ChatRequestSettings settings = new() { MaxTokens = 100 };
            var warStoryOracale = kernel.CreateFunctionFromPrompt(prompt, settings);

            var questions = new[]
            {
                "What is the name of the llama that is going into war?",
                "Summerize the Lluhay war story.",
                "Why is Lluhay going to war against humans?",
                "Who is Lluhay?",
                "What animal is Lluhay?",
            };

            foreach (var question in questions)
            {
                await AskOracale(question, kernel, warStoryOracale);
            }
        }

        private static async Task AskOracale(
            string question, 
            Kernel kernel,
            KernelFunction oracale)
        {
            var result = await kernel.InvokeAsync(oracale, new()
            {
                [TextMemoryPlugin.InputParam] = question,
                [TextMemoryPlugin.CollectionParam] = MemoryCollectionName,
                [TextMemoryPlugin.LimitParam] = "2",
                [TextMemoryPlugin.RelevanceParam] = "0.09",
            }, CancellationToken.None);
            Console.WriteLine(question);
            Console.WriteLine($"Answer: {result.GetValue<string>()}");
        }
    }
}