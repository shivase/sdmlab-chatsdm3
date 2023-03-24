import os
from datetime import datetime
import requests
from slack_bolt import App
from dotenv import load_dotenv

from conversation_agent import ConversationAgent
from qa_agent import QAAgent

load_dotenv()
SLACK_BOT_USER_TOKEN=os.environ.get("SLACK_BOT_USER_TOKEN")
SLACK_BOT_TOKEN=os.environ.get("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET=os.environ.get("SLACK_SIGNING_SECRET")
DOCUMENT_PATH=os.environ.get("DOCUMENT_PATH")

#agent = ConversationAgent()
agent = QAAgent();

def process_event(event, say, memory_key):
    text: str = event['text']
    thread_ts = event.get("thread_ts", None) or event["ts"]

    if text.startswith('initial'):
        agent.set_prefix(text)
        say(text='system prefix定義を設定しました。反映させるには会話をresetで終了して下さい',  thread_ts=thread_ts)
    elif text == 'reset':
        agent.delete(memory_key)
        say(text='会話をリセットしました',  thread_ts=thread_ts)
    else:
        try:
            executor = agent.get_executor(memory_key)
            res = executor({"question": text , "chat_history": []})
            say(text=res['answer'], thread_ts=thread_ts)
        except Exception as error:
            print(error)
            say(text=f"Something Wrong Happened : {error}", thread_ts=thread_ts)

def download_from_slack(file_name: str, download_url: str, auth: str) -> str:
    download_file = requests.get(
        download_url,
        timeout=30,
        allow_redirects=True,
        headers={"Authorization": f"Bearer {auth}"},
        stream=True,
    ).content

    filename = DOCUMENT_PATH + "/" + file_name
    with open(filename, "wb") as file:
        file.write(download_file)

    return filename

if __name__ == "__main__":
    slack = App(
        token=SLACK_BOT_TOKEN,
        signing_secret=SLACK_SIGNING_SECRET
    )

    @slack.event({"type": "message", "subtype": "file_share"})
    def file_share(event, say):
        thread_ts = event.get("thread_ts", None) or event["ts"]
        name = event["files"][0]["name"];
        url = event["files"][0]["url_private_download"]
        filename = download_from_slack(name,url,SLACK_BOT_USER_TOKEN )
        agent.add_document(filename)
        say(text=f'ファイルアップロード: {name} succeeded', thread_ts=thread_ts)

    @slack.event("message")
    def handle_message(body, say):
        event = body['event']
        process_event(event, say, event['user'])

    @slack.event("app_mention")
    def handle_app_mention(body, say):
        event = body['event']
        thread_ts = event.get('thread_ts', event['ts'])
        process_event(event, say, thread_ts)



    slack.start(port=int(os.environ.get("PORT")))