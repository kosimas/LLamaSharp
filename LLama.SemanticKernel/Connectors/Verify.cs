using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace LLamaSharp.SemanticKernel.Connectors;

internal static class Verify
{
    /// <summary>
    /// Equivalent of ArgumentNullException.ThrowIfNull
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void NotNull([NotNull] object? obj, [CallerArgumentExpression("obj")] string? paramName = null)
    {
        if (obj is null)
        {
            ThrowArgumentNullException(paramName);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void NotNullOrWhiteSpace([NotNull] string? str,
        [CallerArgumentExpression("str")] string? paramName = null)
    {
        NotNull(str, paramName);
        if (string.IsNullOrWhiteSpace(str))
        {
            ThrowArgumentWhiteSpaceException(paramName);
        }
    }

    [DoesNotReturn]
    private static void ThrowArgumentInvalidName(string kind, string name, string? paramName) =>
        throw new ArgumentException(
            $"A {kind} can contain only ASCII letters, digits, and underscores: '{name}' is not a valid name.",
            paramName);

    [DoesNotReturn]
    internal static void ThrowArgumentNullException(string? paramName) =>
        throw new ArgumentNullException(paramName);

    [DoesNotReturn]
    internal static void ThrowArgumentWhiteSpaceException(string? paramName) =>
        throw new ArgumentException("The value cannot be an empty string or composed entirely of whitespace.",
            paramName);

    [DoesNotReturn]
    internal static void ThrowArgumentOutOfRangeException<T>(string? paramName, T actualValue, string message) =>
        throw new ArgumentOutOfRangeException(paramName, actualValue, message);
}