using Azure;
using Azure.AI.OpenAI;
using Microsoft.Extensions.Configuration;
using OpenAI.Chat;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

namespace AzureRAGDemo
{
    class Program
    {
        static async Task Main(string[] args)
        {
            // Build configuration
            var configuration = new ConfigurationBuilder()
            .AddJsonFile(Path.Combine(AppContext.BaseDirectory, "appsettings.json"), optional: false, reloadOnChange: true)
            .Build();

            // Load Azure Search settings
            var searchService = configuration["AzureSearch:ServiceUrl"];
            var indexName = configuration["AzureSearch:IndexName"];
            var searchApiVersion = configuration["AzureSearch:ApiVersion"];
            var searchKey = configuration["AzureSearch:ApiKey"];

            // Load Azure OpenAI settings
            var openAiEndpoint = configuration["AzureOpenAI:Endpoint"];
            var embeddingDeployment = configuration["AzureOpenAI:EmbeddingDeployment"];
            var chatDeployment = configuration["AzureOpenAI:ChatDeployment"];
            var openAiKey = configuration["AzureOpenAI:ApiKey"];

            // Create Azure OpenAI client
            AzureOpenAIClient azureClient = new AzureOpenAIClient(
                new Uri(openAiEndpoint),
                new Azure.AzureKeyCredential(openAiKey)
            );

            ChatClient chatClient = azureClient.GetChatClient(chatDeployment);

            var requestOptions = new ChatCompletionOptions()
            {
                MaxOutputTokenCount = 100,
                Temperature = 1.0f,
                TopP = 1.0f,
            };

            List<ChatMessage> messages = new List<ChatMessage>()
            {
                new SystemChatMessage("You are a helpful assistant."),
                new UserChatMessage("I am going to Paris, what should I see?"),
            };

            var response = chatClient.CompleteChatStreaming(messages, requestOptions);
            foreach (StreamingChatCompletionUpdate update in response)
            {
                foreach (ChatMessageContentPart updatePart in update.ContentUpdate)
                {
                    Console.Write(updatePart.Text);
                }
            }
            Console.WriteLine();
        }
    }
}