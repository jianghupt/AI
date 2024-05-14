
import os
import qianfan
import sys

os.environ["QIANFAN_ACCESS_KEY"] = "your accesss_key"
os.environ["QIANFAN_SECRET_KEY"] = "your secret_key"

chat_comp = qianfan.ChatCompletion()

def ABC(str):
      resp = chat_comp.do(model="Yi-34B-Chat", messages=[{"role": "user","content": str}])
      print(resp["body"]['result'])


def main():
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


if __name__ == '__main__':
    main()