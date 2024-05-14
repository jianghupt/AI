using Newtonsoft.Json.Linq;
using RestSharp;
using System.Text;
using System.Text.Json.Serialization;



namespace Web_AOT
{
    public class Program
    {
        public static string llama(string question)
        {
            var client = new RestClient("https://api.atomecho.cn/v1/chat/completions");
            var request = new RestRequest("https://api.atomecho.cn/v1/chat/completions", Method.Post);
            request.AddHeader("Authorization", "Bearer sk-00d4841ba02fc0484cf286d2ff39d824");
            request.AddHeader("Content-Type", "application/json");
            request.AddHeader("Accept", "*/*");
            request.AddHeader("Host", "api.atomecho.cn");
            request.AddHeader("Connection", "keep-alive");
            var body = @"{
" + "\n" +
            @"    ""model"": ""Atom-7B-Chat"",
" + "\n" +
            @"    ""messages"": [
" + "\n" +
            @"        {
" + "\n" +
            @"            ""role"": ""user"",
" + "\n" +
            @"            ""content"":""+" + question + @"+""
" + "\n" +
            @"        }
" + "\n" +
            @"    ],
" + "\n" +
            @"    ""temperature"": 0.3,
" + "\n" +
            @"    ""stream"": true
" + "\n" +
            @"}";
            request.AddParameter("application/json", body, ParameterType.RequestBody);
            RestResponse response = client.Execute(request);

            string str = response.Content;
            string[] str1 = str.Split("data");

            StringBuilder addstr = new StringBuilder();
            for (int i = 0; i < str1.Length - 1; i++)
            {
                try
                {
                    string gg = str1[i].Substring(str1[i].IndexOf("content") + 10, (str1[i].IndexOf("usage") - str1[i].IndexOf("content")) - 16);
                    if (!gg.Contains("finish_reason"))
                        addstr.Append(gg);
                }
                catch
                {
                }
            }
            return addstr.ToString();
        }

        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateSlimBuilder(args);

            builder.Services.ConfigureHttpJsonOptions(options =>
            {
                options.SerializerOptions.TypeInfoResolverChain.Insert(0, AppJsonSerializerContext.Default);
            });

            var app = builder.Build();

            var sampleTodos = new Todo[] {
                new(1, "Walk the dog"),
                new(2, "Do the dishes", DateOnly.FromDateTime(DateTime.Now)),
                new(3, "Do the laundry", DateOnly.FromDateTime(DateTime.Now.AddDays(1))),
                new(4, "Clean the bathroom"),
                new(5, "Clean the car", DateOnly.FromDateTime(DateTime.Now.AddDays(2)))
            };

            var todosApi = app.MapGroup("/llama");
            todosApi.MapGet("/", () => "你好，欢迎来到llama大模型对话");
            //todosApi.MapGet("/{id}", (int id) =>
            //    sampleTodos.FirstOrDefault(a => a.Id == id) is { } todo
            //        ? Results.Ok(todo)
            //        : Results.NotFound());

            todosApi.MapGet("/{question}", (string question) =>
            llama(question)  
            );

            app.Run();
        }
    }

    public record Todo(int Id, string? Title, DateOnly? DueBy = null, bool IsComplete = false);

    [JsonSerializable(typeof(Todo[]))]
    internal partial class AppJsonSerializerContext : JsonSerializerContext
    {

    }
}
