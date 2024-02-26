using LLama;
using LLama.Abstractions;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Services;
using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using LLamaSharp.SemanticKernel.Connectors.LLama;
using static LLama.LLamaTransforms;
using NullReferenceException = System.NullReferenceException;

namespace LLamaSharp.SemanticKernel.ChatCompletion;

/// <summary>
/// LLamaSharp ChatCompletion
/// </summary>
public sealed class LLamaSharpChatCompletion : IChatCompletionService
{
    private readonly StatelessExecutor _model;
    private ChatRequestSettings defaultRequestSettings;
    private readonly IHistoryTransform historyTransform;
    private readonly ITextStreamTransform outputTransform;

    private readonly Dictionary<string, object?> _attributes = new();

    public IReadOnlyDictionary<string, object?> Attributes => this._attributes;

    static ChatRequestSettings GetDefaultSettings()
    {
        return new ChatRequestSettings
        {
            MaxTokens = 256,
            Temperature = 0,
            TopP = 0,
            StopSequences = new List<string>()
        };
    }

    public LLamaSharpChatCompletion(StatelessExecutor model,
        ChatRequestSettings? defaultRequestSettings = default,
        IHistoryTransform? historyTransform = null,
        ITextStreamTransform? outputTransform = null)
    {
        this._model = model;
        this.defaultRequestSettings = defaultRequestSettings ?? GetDefaultSettings();
        this.historyTransform = historyTransform ?? new HistoryTransform();
        this.outputTransform = outputTransform ?? new KeywordTextOutputStreamTransform(new[]
        {
            $"{LLama.Common.AuthorRole.User}:",
            $"{LLama.Common.AuthorRole.Assistant}:",
            $"{LLama.Common.AuthorRole.System}:"
        });
    }

    public ChatHistory CreateNewChat(string? instructions = "")
    {
        var history = new ChatHistory();

        if (instructions != null && !string.IsNullOrEmpty(instructions))
        {
            history.AddSystemMessage(instructions);
        }

        return history;
    }

    /// <exception cref="ArgumentNullException"></exception>
    /// <inheritdoc/>
    public async Task<IReadOnlyList<ChatMessageContent>> GetChatMessageContentsAsync(
        ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings = null,
        Kernel? kernel = null,
        CancellationToken cancellationToken = default)
    {
        var settings = executionSettings != null
            ? ChatRequestSettings.FromRequestSettings(executionSettings)
            : defaultRequestSettings;

        var autoInvoke = kernel is not null && settings.AutoInvoke == true;

        if (!autoInvoke || kernel is null)
        {
            var prompt = historyTransform.HistoryToText(chatHistory.ToLLamaSharpChatHistory());
            var result = _model.InferAsync(prompt, settings.ToLLamaSharpInferenceParams(), cancellationToken);
            var output = outputTransform.TransformAsync(result);

            var sb = new StringBuilder();
            await foreach (var token in output)
            {
                sb.Append(token);
            }

            return new List<ChatMessageContent> { new(AuthorRole.Assistant, sb.ToString()) }.AsReadOnly();
        }

        var promptFunctionCall =
            (historyTransform as HistoryTransform)?.HistoryToTextFC(chatHistory.ToLLamaSharpChatHistory())!;
        var resultFunctionCall =
            _model.InferAsync(promptFunctionCall, settings.ToLLamaSharpInferenceParams(), cancellationToken);
        var outputFunctionCall = outputTransform.TransformAsync(resultFunctionCall);
        var sbFunctionCall = new StringBuilder();

        await foreach (var token in outputFunctionCall)
        {
            sbFunctionCall.Append(token);
        }

        // Add {" to the start of the string as an hack for better results.
        var functionCallString = sbFunctionCall.ToString();
        if (!functionCallString.StartsWith("{")) 
            functionCallString = "{\"" + functionCallString;

        var historyToAppend = new List<ChatMessageContent>
            { new(new AuthorRole("FunctionCall"), functionCallString) };

        try
        {
            var parsedJson = JsonSerializer.Deserialize<LLamaFunctionCall>(functionCallString);
            if (parsedJson == null) throw new NullReferenceException();

            KernelFunction? function;
            kernel.Plugins.TryGetFunction(pluginName: null, functionName: parsedJson.name, out function);
            if (function is null) throw new NullReferenceException();

            var arguments = new KernelArguments(parsedJson.arguments);

            var functionResult = await kernel.InvokeAsync(function, arguments, cancellationToken);
            historyToAppend.Add(new ChatMessageContent(new AuthorRole("FunctionResult"),
                functionResult.GetValue<object>().ToString()));
            chatHistory.AddRange(historyToAppend);
            var promptFunctionResult = historyTransform.HistoryToText(chatHistory.ToLLamaSharpChatHistory());
            var resultFunctionResult = _model.InferAsync(promptFunctionResult, settings.ToLLamaSharpInferenceParams(),
                cancellationToken);
            var outputFunctionResult = outputTransform.TransformAsync(resultFunctionResult);

            var sbFunctionResult = new StringBuilder();
            await foreach (var token in outputFunctionResult)
            {
                sbFunctionResult.Append(token);
            }

            return new List<ChatMessageContent> { new(AuthorRole.Assistant, sbFunctionResult.ToString()) }.AsReadOnly();
        }
        catch (Exception e)
        {
            Debug.WriteLine(e);
            var lastChatMessage = historyToAppend[^1];
            Debug.WriteLine(lastChatMessage.Content);
            switch (lastChatMessage.Role.ToString())
            {
                case "FunctionCall":
                case "function_call":
                    historyToAppend.Remove(historyToAppend[^1]);
                    break;
                case "FunctionResult":
                case "function_result":
                    historyToAppend.RemoveRange(historyToAppend.Count - 2, 2);
                    break;
            }

            var prompt = historyTransform.HistoryToText(chatHistory.ToLLamaSharpChatHistory());
            var result = _model.InferAsync(prompt, settings.ToLLamaSharpInferenceParams(), cancellationToken);
            var output = outputTransform.TransformAsync(result);

            var sb = new StringBuilder();
            await foreach (var token in output)
            {
                sb.Append(token);
            }

            historyToAppend.Add(new ChatMessageContent(AuthorRole.Assistant, sb.ToString()));
            return historyToAppend.AsReadOnly();
        }
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<StreamingChatMessageContent> GetStreamingChatMessageContentsAsync(
        ChatHistory chatHistory,
        PromptExecutionSettings? executionSettings = null,
        Kernel? kernel = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var settings = executionSettings != null
            ? ChatRequestSettings.FromRequestSettings(executionSettings)
            : defaultRequestSettings;
        var prompt = historyTransform.HistoryToText(chatHistory.ToLLamaSharpChatHistory());

        var result = _model.InferAsync(prompt, settings.ToLLamaSharpInferenceParams(), cancellationToken);

        var output = outputTransform.TransformAsync(result);

        await foreach (var token in output)
        {
            yield return new StreamingChatMessageContent(AuthorRole.Assistant, token);
        }
    }
}