using LLama.Common;
using System.Text;
using static LLama.LLamaTransforms;

namespace LLamaSharp.SemanticKernel.ChatCompletion;

/// <summary>
/// Default HistoryTransform Patch
/// </summary>
public class HistoryTransform : DefaultHistoryTransform
{
    /// <inheritdoc/>
    public override string HistoryToText(global::LLama.Common.ChatHistory history)
    {
        return base.HistoryToText(history) + $"{AuthorRole.Assistant}: ";
    }
    
    public string HistoryToTextFC(global::LLama.Common.ChatHistory history)
    {
        // Add {" to the start of the string as an hack for better results.
        return base.HistoryToText(history) + $"{AuthorRole.FunctionCall}: " + "{\"";
    }
}
