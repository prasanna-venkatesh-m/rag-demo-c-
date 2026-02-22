using Azure;
using Azure.AI.OpenAI;
using Azure.Search.Documents;
using Azure.Search.Documents.Models;
using Microsoft.Extensions.Configuration;
using Microsoft.VisualBasic.FileIO;
using OpenAI.Chat;
using System.Text;
using System.Text.Json;
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
            var configuration = new ConfigurationBuilder()
                .AddJsonFile("appsettings.json")
                .Build();

            // ==============================
            // AZURE SEARCH SETTINGS
            // ==============================
            string searchService = configuration["AzureSearch:ServiceUrl"];
            string indexName = configuration["AzureSearch:IndexName"];
            string searchKey = configuration["AzureSearch:ApiKey"];

            // ==============================
            // AZURE OPENAI SETTINGS
            // ==============================
            string openAiEndpoint = configuration["AzureOpenAI:Endpoint"];
            string embeddingDeployment = configuration["AzureOpenAI:EmbeddingDeployment"];
            string chatDeployment = configuration["AzureOpenAI:ChatDeployment"];
            string openAiKey = configuration["AzureOpenAI:ApiKey"];

            // Create Azure OpenAI client
            AzureOpenAIClient azureClient = new AzureOpenAIClient(
                new Uri(openAiEndpoint),
                new AzureKeyCredential(openAiKey)
            );

            var embeddingClient = azureClient.GetEmbeddingClient(embeddingDeployment);
            var chatClient = azureClient.GetChatClient(chatDeployment);

            Console.WriteLine("Enter your question:");
            string query = Console.ReadLine();

            // ==============================
            // 1️⃣ CREATE EMBEDDING
            // ==============================
            var embeddingResponse = await embeddingClient.GenerateEmbeddingAsync(query);
            var queryVector = embeddingResponse.Value.ToFloats().ToArray();

            // ==============================
            // 2️⃣ VECTOR SEARCH
            // ==============================

            var searchClient = new SearchClient(
                new Uri(searchService),
                indexName,
                new AzureKeyCredential(searchKey)
            );

            var vectorQuery = new VectorizedQuery(queryVector)
            {
                KNearestNeighborsCount = 3,
                Fields = { "text_vector" }
            };

            var searchOptions = new SearchOptions
            {
                Size = 3,
            };

            searchOptions.VectorSearch = new VectorSearchOptions();
            searchOptions.VectorSearch.Queries.Add(vectorQuery);

            var response = await searchClient.SearchAsync<SearchDocument>(
                searchText: null,
                searchOptions
            );

            List<string> documents = new List<string>();
            List<(string Title, string ParentId)> citations = new();

            await foreach (SearchResult<SearchDocument> result in response.Value.GetResultsAsync())
            {
                var doc = result.Document;

                if (doc.ContainsKey("chunk"))
                    documents.Add(doc["chunk"]?.ToString());

                citations.Add((
                    doc.ContainsKey("title") ? doc["title"]?.ToString() : "No Title",
                    doc.ContainsKey("parent_id") ? doc["parent_id"]?.ToString() : ""
                ));
            }

            string contextText = documents.Count > 0
                ? string.Join("\n\n", documents)
                : "No documents found.";

            // ==============================
            // 3️⃣ BUILD PROMPT
            // ==============================
            string prompt = $@"
            You are a helpful computer troubleshooting assistant.

            Using ONLY the DOCUMENTS provided below, answer the user's question clearly and step-by-step.
            Do NOT copy documents verbatim.

            If the answer cannot be found in the documents, respond exactly with:
            ""Please contact agent.""

            DOCUMENTS:
            {contextText}

            USER QUESTION:
            {query}
            ";

            // ==============================
            // 4️⃣ GPT RESPONSE
            // ==============================
            var messages = new List<ChatMessage>
            {
                new UserChatMessage(prompt)
            };

            Console.WriteLine("\nResponse:\n");

            var streamingResponse = chatClient.CompleteChatStreamingAsync(
                messages,
                new ChatCompletionOptions
                {
                    Temperature = 0.3f,
                    MaxOutputTokenCount = 800
                });

            await foreach (StreamingChatCompletionUpdate update in streamingResponse)
            {
                foreach (ChatMessageContentPart contentPart in update.ContentUpdate)
                {
                    Console.Write(contentPart.Text);
                }
            }

            Console.WriteLine();

            // ==============================
            // 5️⃣ PRINT CITATIONS
            // ==============================
            if (citations.Count > 0)
            {
                Console.WriteLine("\nCitations:");
                foreach (var c in citations)
                {
                    Console.WriteLine($"Title: {c.Title}");
                    Console.WriteLine($"Parent ID: {c.ParentId}");
                    Console.WriteLine();
                }
            }
        }
    }
}