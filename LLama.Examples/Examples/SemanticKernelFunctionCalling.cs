using LLama;
using LLama.Common;
using LLamaSharp.SemanticKernel.ChatCompletion;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.KernelMemory;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Newtonsoft.Json.Linq;
using ChatHistory = Microsoft.SemanticKernel.ChatCompletion.ChatHistory;

namespace LLama.Examples.Examples;

public class SemanticKernelFunctionCalling
{
    public static async Task Run()
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine("Use this model: " +
                          "https://huggingface.co/Trelis/Llama-2-7b-chat-hf-function-calling-v3");
        var modelPath = UserSettings.GetModelPath();

        var parameters = new ModelParams(modelPath);
        var weights = LLamaWeights.LoadFromFile(parameters);
        var executor = new StatelessExecutor(weights, parameters);

        // Create kernel
        var builder = Kernel.CreateBuilder();
        builder.Services.AddSingleton<IChatCompletionService>(new LLamaSharpChatCompletion(executor));
        builder.Services.AddLogging(c => c.AddDebug().SetMinimumLevel(LogLevel.Trace));
        var plugin = builder.Plugins.AddFromType<MathPlugin>();
        var kernel = builder.Build();

        // Create chat history
        ChatHistory history = [];

        var plugins = kernel.Plugins.GetFunctionsMetadata();
        var functionsPrompt = CreateFunctionsMetaObject(plugins);

        // Prompt for llama hf function calling v3
        var prompt = $"""
                      [INST] You have access to the following functions. Use them if required:
                      {functionsPrompt}
                      """;

        history.AddSystemMessage(prompt);

        // Get chat completion service
        var chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();

        // Start the conversation
        while (true)
        {
            // Get user input
            Console.Write("User > ");
            var userMessage = Console.ReadLine()!;
            history.AddUserMessage(userMessage);

            ChatRequestSettings llamaSettings = new()
            {
                AutoInvoke = true,
            };

            var result = chatCompletionService.GetStreamingChatMessageContentsAsync(
                history,
                executionSettings: llamaSettings,
                kernel: kernel);

            Console.Write("Assistant > ");
            var lastMessage = "";
            await foreach (var content in result)
            {
                Console.Write(content.ToString());
                lastMessage += content.ToString();
            }

            // var lastMessage = result.Last().Content;
            // Console.WriteLine(lastMessage);

            Console.WriteLine();

            // Add the message from the agent to the chat history
            history.AddAssistantMessage(lastMessage);
        }
    }

    private static JToken? CreateFunctionsMetaObject(IList<KernelFunctionMetadata> plugins)
    {
        if (plugins.Count < 1) return null;
        if (plugins.Count == 1) return CreateFunctionMetaObject(plugins[0]);
        
        JArray promptFunctions = [];
        foreach (var plugin in plugins)
        {
            var pluginFunctionWrapper = CreateFunctionMetaObject(plugin);
            promptFunctions.Add(pluginFunctionWrapper);
        }

        return promptFunctions;
    }

    private static JObject CreateFunctionMetaObject(KernelFunctionMetadata plugin)
    {
        var pluginFunctionWrapper = new JObject()
        {
            { "type", "function" },
        };

        var pluginFunction = new JObject()
        {
            { "name", plugin.Name },
            { "description", plugin.Description },
        };

        var pluginFunctionParameters = new JObject()
        {
            { "type", "object" },
        };
        var pluginProperties = new JObject();
        foreach (var parameter in plugin.Parameters)
        {
            var property = new JObject()
            {
                { "type", parameter.ParameterType?.ToString() },
                { "description", parameter.Description },
            };

            pluginProperties.Add(parameter.Name, property);
        }

        pluginFunctionParameters.Add("properties", pluginProperties);
        pluginFunction.Add("parameters", pluginFunctionParameters);
        pluginFunctionWrapper.Add("function", pluginFunction);

        return pluginFunctionWrapper;
    }
}