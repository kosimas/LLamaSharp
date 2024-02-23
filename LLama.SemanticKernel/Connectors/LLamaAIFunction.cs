using System.Text.Json;
using System.Text.Json.Serialization;
using Json.Schema;
using Json.Schema.Generation;
using Microsoft.SemanticKernel;
using Azure.AI.OpenAI;

namespace LLamaSharp.SemanticKernel.Connectors.LLama;

public sealed class LLamaFunctionParameter
{
    internal LLamaFunctionParameter(string? name, string? description, bool isRequired, Type? parameterType,
        KernelJsonSchema? schema)
    {
        this.Name = name ?? string.Empty;
        this.Description = description ?? string.Empty;
        this.IsRequired = isRequired;
        this.ParameterType = parameterType;
        this.Schema = schema;
    }

    /// <summary>Gets the name of the parameter.</summary>
    public string Name { get; }

    /// <summary>Gets a description of the parameter.</summary>
    public string Description { get; }

    /// <summary>Gets whether the parameter is required vs optional.</summary>
    public bool IsRequired { get; }

    /// <summary>Gets the <see cref="Type"/> of the parameter, if known.</summary>
    public Type? ParameterType { get; }

    /// <summary>Gets a JSON schema for the parameter, if known.</summary>
    public KernelJsonSchema? Schema { get; }
}

/// <summary>
/// Represents a function return parameter that can be returned by a tool call to LLama.
/// </summary>
public sealed class LLamaFunctionReturnParameter
{
    internal LLamaFunctionReturnParameter(string? description, Type? parameterType, KernelJsonSchema? schema)
    {
        this.Description = description ?? string.Empty;
        this.Schema = schema;
        this.ParameterType = parameterType;
    }

    /// <summary>Gets a description of the return parameter.</summary>
    public string Description { get; }

    /// <summary>Gets the <see cref="Type"/> of the return parameter, if known.</summary>
    public Type? ParameterType { get; }

    /// <summary>Gets a JSON schema for the return parameter, if known.</summary>
    public KernelJsonSchema? Schema { get; }
}

/// <summary>
/// Represents a function that can be passed to the LLama API
/// </summary>
public sealed class LLamaFunction
{
    /// <summary>
    /// Cached <see cref="BinaryData"/> storing the JSON for a function with no parameters.
    /// </summary>
    /// <remarks>
    /// This is an optimization to avoid serializing the same JSON Schema over and over again
    /// for this relatively common case.
    /// </remarks>
    private static readonly BinaryData s_zeroFunctionParametersSchema =
        new("{\"type\":\"object\",\"required\":[],\"properties\":{}}");

    /// <summary>
    /// Cached schema for a descriptionless string.
    /// </summary>
    private static readonly KernelJsonSchema s_stringNoDescriptionSchema =
        KernelJsonSchema.Parse("{\"type\":\"string\"}");

    /// <summary>Initializes the LLamaFunction.</summary>
    internal LLamaFunction(
        string? pluginName,
        string functionName,
        string? description,
        IReadOnlyList<LLamaFunctionParameter>? parameters,
        LLamaFunctionReturnParameter? returnParameter)
    {
        Verify.NotNullOrWhiteSpace(functionName);

        this.PluginName = pluginName;
        this.FunctionName = functionName;
        this.Description = description;
        this.Parameters = parameters;
        this.ReturnParameter = returnParameter;
    }

    /// <summary>Gets the separator used between the plugin name and the function name, if a plugin name is present.</summary>
    public static string NameSeparator => "_";

    /// <summary>Gets the name of the plugin with which the function is associated, if any.</summary>
    public string? PluginName { get; }

    /// <summary>Gets the name of the function.</summary>
    public string FunctionName { get; }

    /// <summary>Gets the fully-qualified name of the function.</summary>
    /// <remarks>
    /// This is the concatenation of the <see cref="PluginName"/> and the <see cref="FunctionName"/>,
    /// separated by <see cref="NameSeparator"/>. If there is no <see cref="PluginName"/>, this is
    /// the same as <see cref="FunctionName"/>.
    /// </remarks>
    public string FullyQualifiedName =>
        string.IsNullOrEmpty(this.PluginName)
            ? this.FunctionName
            : $"{this.PluginName}{NameSeparator}{this.FunctionName}";

    /// <summary>Gets a description of the function.</summary>
    public string? Description { get; }

    /// <summary>Gets a list of parameters to the function, if any.</summary>
    public IReadOnlyList<LLamaFunctionParameter>? Parameters { get; }

    /// <summary>Gets the return parameter of the function, if any.</summary>
    public LLamaFunctionReturnParameter? ReturnParameter { get; }

    /// <summary>
    /// Converts the <see cref="LLamaFunction"/> representation to the Azure SDK's
    /// <see cref="FunctionDefinition"/> representation.
    /// </summary>
    /// <returns>A <see cref="FunctionDefinition"/> containing all the function information.</returns>
    public FunctionDefinition ToFunctionDefinition()
    {
        BinaryData resultParameters = s_zeroFunctionParametersSchema;

        IReadOnlyList<LLamaFunctionParameter>? parameters = this.Parameters;
        if (parameters is { Count: > 0 })
        {
            var properties = new Dictionary<string, KernelJsonSchema>();
            var required = new List<string>();

            for (int i = 0; i < parameters.Count; i++)
            {
                var parameter = parameters[i];
                properties.Add(parameter.Name,
                    parameter.Schema ?? GetDefaultSchemaForTypelessParameter(parameter.Description));
                if (parameter.IsRequired)
                {
                    required.Add(parameter.Name);
                }
            }

            resultParameters = BinaryData.FromObjectAsJson(new
            {
                type = "object",
                required,
                properties,
            });
        }

        return new FunctionDefinition
        {
            Name = this.FullyQualifiedName,
            Description = this.Description,
            Parameters = resultParameters,
        };
    }

    /// <summary>Gets a <see cref="KernelJsonSchema"/> for a typeless parameter with the specified description, defaulting to typeof(string)</summary>
    private static KernelJsonSchema GetDefaultSchemaForTypelessParameter(string? description)
    {
        // If there's a description, incorporate it.
        if (!string.IsNullOrWhiteSpace(description))
        {
            return KernelJsonSchema.Parse(JsonSerializer.Serialize(
                new JsonSchemaBuilder()
                    .FromType(typeof(string))
                    .Description(description!)
                    .Build()));
        }

        // Otherwise, we can use a cached schema for a string with no description.
        return s_stringNoDescriptionSchema;
    }
}

public sealed class LLamaFunctionCall
{
    public string name { get; set; }
    
    [JsonConverter(typeof(DictionaryStringObjectConverter))]
    public Dictionary<string, object?> arguments { get; set; }
}

public class DictionaryStringObjectConverter : JsonConverter<Dictionary<string, object?>>
{
    public override Dictionary<string, object?> Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType != JsonTokenType.StartObject)
        {
            throw new JsonException();
        }

        Dictionary<string, object?> dictionary = new Dictionary<string, object?>();
        while (reader.Read())
        {
            if (reader.TokenType == JsonTokenType.EndObject)
            {
                return dictionary;
            }

            if (reader.TokenType != JsonTokenType.PropertyName)
            {
                throw new JsonException();
            }

            string propertyName = reader.GetString();
            reader.Read();
            object? propertyValue;
            switch (reader.TokenType)
            {
                case JsonTokenType.String:
                    propertyValue = reader.GetString();
                    break;
                case JsonTokenType.Number:
                    propertyValue = reader.TryGetInt64(out long longValue) ? longValue : reader.GetDouble();
                    break;
                default:
                    throw new JsonException($"Unsupported token type: {reader.TokenType}");
            }
            dictionary[propertyName] = propertyValue;
        }

        throw new JsonException("Unclosed object at end of JSON.");
    }

    public override void Write(Utf8JsonWriter writer, Dictionary<string, object?> value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();
        foreach (var kvp in value)
        {
            writer.WritePropertyName(kvp.Key);
            JsonSerializer.Serialize(writer, kvp.Value, options);
        }
        writer.WriteEndObject();
    }
}