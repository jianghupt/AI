from openai import OpenAI
import sys

def ABC(str):
      client = OpenAI(api_key="sk-your atom key",base_url="https://api.atomecho.cn/v1",)
      completion = client.chat.completions.create(model="Llama3-Chinese-8B-Instruct",messages=[{"role":"user", "content": str}],temperature=0.3,)
      print(completion.choices[0].message.content)
      return

print("欢迎使用llama3大模型对话")
while(1):
 try:
   print("\r\n")
   user_input = input("Please type something and press enter: ")
   if user_input =="exit":
      exit(0)
   if len(user_input)!=0:
      ABC(user_input)
   else:
      print("请输入参数")
 except KeyboardInterrupt:
    print("\r\n")
    sys.exit()