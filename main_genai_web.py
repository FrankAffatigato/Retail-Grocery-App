import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI
from langchain import hub
import pandas as pd
import altair as alt
import os

# ---------------- Set Page Config First ----------------
st.set_page_config(page_title="Retail GenAI Assistant", layout="wide")

# Inject custom CSS for green outline on input
st.markdown("""
    <style>
    div[data-baseweb="input"] > div {
        border: 1px solid #ccc;
        border-radius: 6px;
        padding: 0.5rem;
    }
    div[data-baseweb="input"]:focus-within {
        border: 2px solid green !important;
        box-shadow: 0 0 0 0.1rem rgba(0, 128, 0, 0.25);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- Load Environment and Authentication ----------------
load_dotenv()
USERNAME = os.getenv("APP_USERNAME")
PASSWORD = os.getenv("APP_PASSWORD")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login_container = st.empty()
    with login_container.form("login_form"):
        st.subheader("🔐 Login Required")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if user == USERNAME and pwd == PASSWORD:
                st.session_state.authenticated = True
                login_container.empty()
            else:
                st.error("Invalid credentials.")
    if not st.session_state.authenticated:
        st.stop()

# ---------------- App Starts ----------------
st.title("🛍️ Retail Data Chatbot + Dashboard")

# Load static CSV from project repo
csv_path = "sample_retail_data.csv"
if not os.path.exists(csv_path):
    st.error("CSV file not found. Please make sure 'sample_retail_data.csv' is in the project folder.")
    st.stop()

df = pd.read_csv(csv_path, parse_dates=['Date'])

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Prepare tools
instructions = """You are an agent designed to write and execute python code to answer questions.
You have access to a python REPL, which you can use to execute python code.
If you get an error, debug your code and try again.
Only use the output of your code to answer the question. 
You might know the answer without running any code, but you should still run the code to get the answer.
If it does not seem like you can write code to answer the question, just return \"I don't know\" as the answer.
"""

base_prompt = hub.pull("langchain-ai/react-agent-template")
prompt = base_prompt.partial(instructions=instructions)

python_agent = create_react_agent(
    prompt=prompt,
    llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
    tools=[PythonREPLTool()],
)
python_agent_executor = AgentExecutor(agent=python_agent, tools=[PythonREPLTool()], verbose=True)

csv_agent_executor = create_csv_agent(
    llm=ChatOpenAI(temperature=0, model="gpt-4"),
    path=csv_path,
    verbose=True,
    allow_dangerous_code=True
)

# Router tools
tools = [
    Tool(
        name="Python Agent",
        func=lambda x: python_agent_executor.invoke({"input": x}),
        description="""Use this when you need to transform natural language to python and execute the python code,
                      returning the results of the code execution. DOES NOT ACCEPT CODE AS INPUT""",
    ),
    Tool(
        name="CSV Agent",
        func=csv_agent_executor.invoke,
        description="""Useful when you need to answer questions over the uploaded retail CSV file,
                     takes an input with the full question and returns the result after running pandas code""",
    ),
]

# Grand agent
grand_agent = create_react_agent(
    prompt=base_prompt.partial(instructions=""),
    llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
    tools=tools,
)
grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

# -------------------- Chat Section --------------------
st.subheader("💬 Chat with the Assistant")
user_input = st.text_input("Ask a question about the data", placeholder="e.g. What was the total revenue last month?", key="chat_input", label_visibility="visible")
if user_input:
    with st.spinner("Thinking..."):
        response = grand_agent_executor.invoke({"input": user_input})
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", response["output"]))

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)

# -------------------- Dashboard Section --------------------
st.markdown("---")
st.header("📊 Retail Dashboard")

# Sidebar Filters
date_range = st.sidebar.date_input("Select Date Range", [df["Date"].min(), df["Date"].max()])
category_filter = st.sidebar.selectbox("Select Category", options=["All"] + sorted(df['Category'].unique().tolist()))
store_filter = st.sidebar.multiselect("Select Stores", options=sorted(df['Store'].unique()), default=df['Store'].unique())

# Apply filters
filtered_df = df.copy()
if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[(filtered_df['Date'] >= pd.to_datetime(start_date)) &
                              (filtered_df['Date'] <= pd.to_datetime(end_date))]
if category_filter != "All":
    filtered_df = filtered_df[filtered_df['Category'] == category_filter]
if store_filter:
    filtered_df = filtered_df[filtered_df['Store'].isin(store_filter)]

# KPIs
total_revenue = filtered_df['Revenue'].sum()
total_units = filtered_df['Units Sold'].sum()
total_profit = (filtered_df['Revenue'] - filtered_df['Cost']).sum()

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("💰 Total Revenue", f"${total_revenue:,.0f}")
kpi2.metric("📦 Units Sold", f"{total_units:,}")
kpi3.metric("📈 Gross Profit", f"${total_profit:,.0f}")

# Charts
st.markdown("### 📈 Revenue Over Time")
revenue_trend = filtered_df.groupby("Date")["Revenue"].sum().reset_index()
st.altair_chart(
    alt.Chart(revenue_trend).mark_line().encode(
        x="Date:T",
        y="Revenue:Q"
    ).properties(width=800, height=300),
    use_container_width=True
)

st.markdown("### 🧾 Revenue by Category")
category_chart = filtered_df.groupby("Category")["Revenue"].sum().reset_index()
st.bar_chart(category_chart.set_index("Category"))

st.markdown("### 📍 Revenue by Region")
region_chart = filtered_df.groupby("Region")["Revenue"].sum().reset_index()
st.bar_chart(region_chart.set_index("Region"))

with st.expander("🔍 View Raw Data"):
    st.dataframe(filtered_df.reset_index(drop=True))
