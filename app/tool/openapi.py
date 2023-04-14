import time

import PyPDF2
import openai
import pandas as pd
import tiktoken
import xmindparser
from openai.embeddings_utils import distances_from_embeddings
from openai.embeddings_utils import get_embedding, cosine_similarity

from tabulate import tabulate

import config

case_step = 5
openai.api_key = config.ProductionConfig.SECRET_KEY
max_token = 500
model = config.ProductionConfig.OPENAI_MODEL
# Load the tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.encoding_for_model(model)


def pdf2dataframe():
    # test.pdf, mv-v19-1074.pdf
    file_name = 'D:\\PycharmProjects\\openai-quickstart\\app\\tool\\ETSI_TS_124_002_V6_0_0_2004_12.pdf'
    # Open the PDF file
    pdf_file = open(file_name, 'rb')
    # Create a PDF reader object
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    # Get the number of pages in the PDF file
    num_pages = len(pdf_reader.pages)
    pdf_text = ''
    # Loop through each page and extract the text
    for page in range(num_pages):
        # Get the page object
        pdf_page = pdf_reader.pages[page]
        # Extract the text from the page
        pdf_text = pdf_text + pdf_page.extract_text()
    # Remove newline from the text
    pdf_text = remove_newlines(pdf_text)
    # Close the PDF file
    pdf_file.close()
    # Get the file token
    file_token = len(tokenizer.encode(pdf_text))
    # Turn the file text into shorter lines
    shortened = []
    if file_token > max_token:
        shortened += split_into_many(pdf_text)
    else:
        shortened += pdf_text
    df = pd.DataFrame(shortened, columns=['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    # Define a function to get the embeddings for a text
    # def get_embeddings(text):
    #     try:
    #         return openai.Embedding.create(input=text, engine='text-embedding-ada-002')['data'][0]['embedding']
    #     except openai.error.RateLimitError as e:
    #         print("Rate limit reached. Waiting for 1 minute...")
    #         time.sleep(60)
    #         return get_embeddings(text)
    # df['embeddings'] = df['text'].apply(get_embeddings)
    return df


def create_context(question, dataframe, max_len=1800):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    dataframe['distances'] = distances_from_embeddings(q_embeddings, dataframe['embeddings'].values,
                                                       distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in dataframe.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)


# Remove new lines from the text
def remove_newlines(series):
    series = series.replace('\n', ' ')
    series = series.replace('\\n', ' ')
    series = series.replace('  ', ' ')
    series = series.replace('  ', ' ')
    return series


# 将文本拆分为最大数量的标记块的函数
def split_into_many(text, max_tokens=max_token):
    # 将文本拆分成句子
    sentences = text.split('. ')

    # 获取每个句子的标记数
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    chunks = []
    tokens_so_far = 0
    chunk = []

    # 遍历连接在一个元组中的句子和标记
    for sentence, token in zip(sentences, n_tokens):

        # 如果 token 的个数加上当前句子中的 token 的个数大于最大的 token 数，则将 chunk 添加到 chunks 列表中，并重置 chunk 和 tokens
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # 如果当前句子中的token个数大于最大token个数，则转到下一句
        if token > max_tokens:
            continue

        # 否则，将句子添加到块中并将令牌数添加到总数中
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks


def answer_question(
        dataframe,
        question="Am I allowed to publish model outputs to Twitter, without a human review?",
        previous_message=None,
        max_len=1800,
        debug=False,
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    if previous_message is None:
        previous_message = []
    context = create_context(
        question,
        dataframe,
        max_len=max_len,
    )
    # If debugged, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        message = []
        if len(previous_message) != 0:
            message = previous_message
            message.append({"role": "user", "content": f"Answer the question based on the context below, "
                                                       f"and if the question can't be answered based on "
                                                       f"the context, say \"I don't know\"\n\nContext: "
                                                       f"{context}\n\n---\n\nQuestion: {question}"})
        else:
            message.append({"role": "system", "content": "You are a great Patent Researcher, "
                                                         "your name is \\\"AI_Rebot\\\""})
            message.append({"role": "user", "content": """i will give you some context and instruction, and remember 
            your name is \\\"AI_Rebot\\\", Do you understand? """})
            message.append({"role": "system",
                            "content": f"Answer the question based on the context below, and if the question can't be "
                                       f"answered based on the context, say \"I don't know\"\n\nContext: {context}"})
            message.append({"role": "user", "content": f"Question: {question}, and please reply in the "
                                                       f"language of the question, the max length of the "
                                                       f"answer should in 3000 words"})
        # Create a completions using the question and context
        # response = openai.ChatCompletion.create(model=config.ProductionConfig.OPENAI_MODEL, messages=message,
        #                                       max_tokens=int(config.ProductionConfig.OPENAI_MAX_TOKENS),
        #                                       temperature=0.2,
        #                                       timeout=int(config.ProductionConfig.OPENAI_TIMEOUT))

        def search_reviews(df, product_description, n=3, pprint=True):
            embedding = get_embedding(product_description, model='text-embedding-ada-002')
            df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity(x, embedding))
            res = df.sort_values('similarities', ascending=False).head(n)
            return res

        answer = search_reviews(dataframe, message, n=3)

        # answer = response['choices'][0]['message']['content'].replace('\n', '<br>')
        if debug:
            print("answer\n" + answer)
            print("\n\n")
        message.append({"role": "assistant", "content": answer})
        result_tuple = (message, answer)
        return result_tuple
    except Exception as e:
        print(e)
        return ""


def max_level(tree_dict, current_level=0, max_level_number=0):
    if tree_dict:
        current_level += 1
        if current_level > max_level_number:
            max_level_number = current_level
        for tree_dict_item in tree_dict:
            if "topics" in tree_dict_item:
                max_level_number = max_level(tree_dict_item["topics"], current_level, max_level_number)
    return max_level_number


def topic2column(df, tree_dict, titles, max_level_number):
    if tree_dict:
        for tree_dict_item in tree_dict:
            current_titles = titles.copy()
            current_titles.append(tree_dict_item["title"])
            if "topics" in tree_dict_item:
                topic2column(df, tree_dict_item["topics"], current_titles, max_level_number)
            else:
                if len(current_titles) < max_level_number:
                    for i in range(0, max_level_number - len(current_titles)):
                        current_titles.append(" ")
                df.loc[len(df)] = current_titles


def xmind2xls():
    # Drug Search Upgrade.xmind, Synapse PLG 2.0.xmind,专利1.2.xmind,多靶点遗留.xmind,Dashboard 优化.xmind
    xmind_file = 'Synapse PLG 2.0.xmind'
    # 读取 Xmind 文件
    tree = xmindparser.xmind_to_dict(xmind_file)

    # 获取第一个 Topic 的 title 用作 excel 文件名称
    xls_file_name = tree[0]["topic"]["title"] + '.xlsx'

    max_level_number = max_level(tree[0]["topic"]["topics"])

    print("xmind max level is %d" % max_level_number)

    cols = []
    for col in range(max_level_number):
        cols.append("level %d" % col)

    # 构造一个空的 df
    df = pd.DataFrame(columns=cols)

    # 遍历 tree，每次遇到根节点，则往 excel 中增加一行
    titles = []
    topic2column(df, tree[0]["topic"]["topics"], titles, max_level_number)

    # 将 DataFrame 对象写入 Excel 文件
    df.to_excel(xls_file_name, index=False)
    return xls_file_name


def xls2markdown(xls_file):
    df = pd.read_excel(xls_file, header=None)
    num_rows = df.shape[0]

    for row in range(1, num_rows, case_step):
        if (row + case_step) < num_rows:
            df_chunk = df.iloc[row:row + case_step]
        else:
            df_chunk = df.iloc[row:num_rows]

        markdown = tabulate(df_chunk, headers='keys', tablefmt='pipe')

        # Print the Markdown output
        print(markdown)

        message = [{"role": "system", "content": "Now you are a excellent software quality assurance "
                                                 "people, you are good at writing test cases through markdown table"},
                   {"role": "user", "content": "I will give you a context in markdown format, and you will "
                                               "write the test cases, do you understand?"},
                   {"role": "assistant", "content": "Yes, I understand. Please provide me with the context in "
                                                    "Markdown format, and I will write the test cases for you."},
                   {"role": "user", "content": f"Markdown table: \n"
                                               f" {markdown} \n\n "
                                               f"Please write the software "
                                               f"test cases in Chinese, and the reply format should be:\n "
                                               f"[Row Number] [Test cases title] XXXXXXX \n "
                                               f"Test Steps: [Test Steps] \n"
                                               f"Expected Result: [Expected Result] \n"
                                               f"Priority: [Priority] \n\n"
                                               f"Rules:"
                                               f"1. If test steps are more than one line, "
                                               f"please put them into multiple lines \n"
                                               f"2. each test case should less than 150 words \n"
                                               f"3. Test cases title should start with previous [column value], "
                                               f"if there are multiple previous columns, use '-' to separate them  \n"
                                               f"4. Answer in Chinese also \n"
                                               f"5. Each line of the markdown table must be one test case. \n"
                                               f"6. Each 'Test Steps' item must have an 'Expected Result' item to be "
                                               f"mapping to \n "
                                               f"7. 'Priority' only could be '高', '中', '低' \n"
                                               f"8. If the cause is 'search', 'analysis', 'table list', 'data', "
                                               f"'security' "
                                               f"related, the priority should be '高' \n "
                                               f"9. If the case is 'UIUX', 'frontend', 'basic logic' related, "
                                               f"the priority should be '低' \n "
                                               f"10. Except rule 8 and rule 9, for others, the priority should be '中' "
                                               f"\n\n "
                                               f"Examples: \n"
                                               f"if a row of the markdown table is: \n"
                                               f" |1|竞争格局|适应症|临床试验的国家/地区分布| | | | | |, "
                                               f"Test case answer:"
                                               f"[用例1] [药物检索-药物] 临床试验的国家/地区分布 - 检查图表显示正确 \n"
                                               f"测试步骤：\n "
                                               f"1) 检查总览tab，单靶点EGFR的临床试验的国家/地区分布以及显示 \n "
                                               f"2) 检查作用机制tab，单靶点EGFR，EGFR拮抗剂的临床试验的国家/地区分布以及显示 \n "
                                               f"3) 检查多靶点PD1+LAG3 的临床试验的国家/地区分布以及显示 \n"
                                               f"期望结果：\n "
                                               f"1)显示国家/地区的临床分布并且检查美国的临床数量和临床检索 EGFR，过滤美国后，需要数量一致\n"
                                               f"2)显示国家/地区的临床分布数据并且检查美国的临床数量和临床检索 EGFR，过滤美国后，需要数量一致\n"
                                               f"3)显示国家/地区临床分布数据并且检查美国的临床数量和临床检索 PD1+LAG3，过滤美国后，需要数量一致 \n"
                                               f"优先级：\n"
                                               f"高\n\n"
                                               f"Another example, if a row of the markdown table is:\n"
                                               f"|2|公司详情页|专利|公司专利分析|Total patent docs| | | | |"
                                               f"Test case answer: \n"
                                               f"[用例2] [公司详情页-专利-公司专利分析] atents标题后面info中的“Total patent "
                                               f"docs”在勾选与不勾选Include subsidiary时显示不一样 \n "
                                               f"测试步骤：\n "
                                               f"1) 进入Google LLC详情页+ 语言为英文 \n "
                                               f"2) hover在Patents后面的info \n "
                                               f"3) 取消勾选Include subsidiary + 语言切换为中文 \n"
                                               f"期望结果：\n "
                                               f"1)Data Snapshot 后面的 Include subsidiary 默认勾选\n"
                                               f"2)展示：This patent count is grouped by 'one doc per application'. "
                                               f"Total patent docs: 163K\n "
                                               f"3)专利总数按照APNO方式折叠，总专利文档数：128K\n"
                                               f"优先级：\n"
                                               f"高\n\n"
                    }]
        response = openai.ChatCompletion.create(model=config.ProductionConfig.OPENAI_MODEL, messages=message,
                                                max_tokens=int(config.ProductionConfig.OPENAI_MAX_TOKENS),
                                                temperature=0.8,
                                                timeout=int(config.ProductionConfig.OPENAI_TIMEOUT))
        answer = response['choices'][0]['message']['content'].replace('\n', '<br>')
        print(answer)
