from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage

import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

import pandas as pd
import random
from datetime import datetime

load_dotenv()

# Importing dataframes


transac_df = pd.read_csv("C:/python/langchain/LangGraph/transactions.csv")
nodal_df = pd.read_csv("C:/python/langchain/LangGraph/nodal.csv")
acc_balance_df = pd.read_csv("C:/python/langchain/LangGraph/acc_balance_create.csv")
data = {
    "Transaction_ID": [],
    "Issue_ID": [],
    "Transaction_Date": [],
    "Amount": [],
    "Receiver_UPI_ID": [],
    "Issue_Type": [],
    "Description": [],
    "Time_stamp":[]
}
complaints_df = pd.DataFrame(data)
nodal_df.head()

# State class


from typing import TypedDict
class State(TypedDict):
    messages: Annotated[list,add_messages]

# LLM Definition


api_key=os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.3
    )

# Defining Tools


# tools
def verify_transaction(trans_id: str, trans_date: str, amount: str, receiver_upi_id: str) -> str:
    """Verify if transaction exists in the system.
   
    Args:
        trans_id (str): Transaction ID
        trans_date (str): Transaction date (YYYY-MM-DD)
        amount (float): Transaction amount
        receiver_upi_id (str): Receiver's UPI ID
   
    Returns:
        str: Verification result
    """
    try:
        # Find matching transaction
        match = transac_df[
            (transac_df['transaction_id'].astype(str) == str(trans_id)) &
            (transac_df['transaction_date'].astype(str) == str(trans_date)) &
            (transac_df['amount'].astype(str) == str(amount)) &
            (transac_df['receiver_id'].astype(str) == str(receiver_upi_id))
        ]
       
        if match.empty:
            return f"✗ Transaction NOT FOUND - ID: {trans_id}"
       
        # ✅ GET SINGLE ROW
        result = match.iloc[0]  # Single row as Series
       
        # ✅ EXTRACT SINGLE VALUES (not .values!)
        status = result['status']          # Single value
        debit_status = result['debit_status']
        credit_status = result['credit_status']
       
        # ✅ COMPARE SINGLE VALUES
        if (str(status).lower() == 'failed' and
            str(debit_status).lower() == 'yes' and
            str(credit_status).lower() == 'no'):
            return f"✓ Transaction VERIFIED - ID: {trans_id}, Amount: {amount}"
        else:
            return f"✗ NOT ELIGIBLE - Status: {status}, Debit: {debit_status}, Credit: {credit_status}"
   
    except Exception as e:
        return f"Error: {str(e)}"

def process_refund(trans_id: str, amount: str) -> str:
    """Process refund by finding customer_id automatically and updating balances.
   
    Flow:
    1. Get trans data from transac_df using trans_id
    2. Find matching record in nodal_df using data from transac
    3. Extract customer_id from nodal_df
    4. Update acc_balance_df for that customer_id
    5. Delete nodal_df record
    """
    try:
        amount_int = int(amount)
       
        print(f"\n[REFUND] Processing refund for {trans_id}, amount ₹{amount_int}")
       
        # ✅ STEP 1: Get transaction from transac_df
        transaction = transac_df[transac_df['transaction_id'].astype(str) == str(trans_id)]
       
        if transaction.empty:
            return f"✗ Transaction {trans_id} not found"
       
        trans = transaction.iloc[0]
        print(f"[TRANS] Found transaction: {trans_id}")
       
        # Get fields that match with nodal_df
        # Assuming nodal_df has similar fields - adjust based on your actual column names
        trans_amount = trans['amount']
        trans_date = trans['transaction_date']
       
        # ✅ STEP 2: Find matching record in nodal_df
        nodal_match = nodal_df[
            (nodal_df['transaction_id'].astype(str) == str(trans_id)) |
            ((nodal_df['amount'].astype(str) == str(trans_amount)))
        ]
       
        if nodal_match.empty:
            return f"✗ No matching record in nodal_df for transaction {trans_id}"
       
        nodal_record = nodal_match.iloc[0]
        nodal_index = nodal_match.index[0]
       
        print(f"[NODAL] Found matching nodal record at index {nodal_index}")
       
        # ✅ STEP 3: Extract customer_id from nodal_df
        customer_id = str(nodal_record['customer_id'])
       
        print(f"[CUSTOMER] Extracted customer_id: {customer_id}")
        print(f"[NODAL RECORD] {nodal_record.to_dict()}")
       
        # ✅ STEP 4: Find and update customer balance in acc_balance_df
        balance_match = acc_balance_df[acc_balance_df['customer_id'].astype(str) == customer_id]
       
        if balance_match.empty:
            return f"✗ Customer {customer_id} not found in acc_balance_df"
       
        balance_index = balance_match.index[0]
        current_balance_before = int(balance_match.iloc[0]['current_balance'])
        current_balance_after = current_balance_before + amount_int
       
        print(f"[BALANCE] Customer {customer_id}")
        print(f"  Before: ₹{current_balance_before}")
        print(f"  Adding: ₹{amount_int}")
        print(f"  After: ₹{current_balance_after}")
       
        # Update balance
        acc_balance_df.loc[balance_index, 'current_balance'] = current_balance_after
       
        # ✅ STEP 5: Delete nodal record
        nodal_df.drop(nodal_index, inplace=True)
        nodal_df.reset_index(drop=True, inplace=True)
       
        print(f"[DELETE] Deleted nodal record (index {nodal_index})")
        print(f"[NODAL] Remaining records: {len(nodal_df)}")
       
        # Create confirmation
        refund_id = f"REFUND_{random.randint(100000, 999999)}"
       
        confirmation = f"""✓ REFUND COMPLETED

Refund ID: {refund_id}
Transaction ID: {trans_id}
Customer ID: {customer_id}
Amount: ₹{amount_int}

Balance Update:
- Before: ₹{current_balance_before}
- After: ₹{current_balance_after}

Actions Completed:
✓ Updated acc_balance_df (Customer {customer_id})
✓ Deleted nodal record

Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
       
        print(f"[SUCCESS] Refund completed")
        return confirmation
   
    except Exception as e:
        import traceback
        return f"✗ Error: {str(e)}\n{traceback.format_exc()}"
   

# Binding Tools


tools = [verify_transaction,process_refund]
system_prompt = """You are a helpful assistant with access to:
1. CSV data tool - use when user asks about data ("select where...", "show data...", "find...",retrieve...)
When user asks about data, use pandas_code tool."""

llm_with_tools = llm.bind_tools(tools, system_prompt=system_prompt)

# Creating Nodes and Edges


from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_core.messages import SystemMessage

def tool_calling_node(state:State):
    """Banking refund agent"""
    system_prompt = """Banking refund assistant.

1. verify_transaction(trans_id, trans_date, amount, receiver_upi_id)
2. If eligible: process_refund(trans_id, amount)
3. Return result with proper description. No questions."""
   
    # ADD system message to messages list
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
   
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

builder = StateGraph(State)
builder.add_node("tool_calling_node",tool_calling_node)
builder.add_node("tools",ToolNode(tools))

builder.add_edge(START,"tool_calling_node")
builder.add_conditional_edges("tool_calling_node",tools_condition)
builder.add_edge("tools","tool_calling_node")

graph = builder.compile()

# Giving input


trans_id = input("Transaction_id: ")
issue_id = random.randint(100001,999999)  
trans_date = input("Transaction_date(DD-MM-YYYY): ")
amount = input("Amount: ")
receiver_UPI_ID = input("Receiver_UPI_ID: ")
issue_type = input("Issue_type: ")
description = input("Description: ")
time_stamp = datetime.now().time()    

if trans_id in str(complaints_df.Transaction_ID):
    print("Complaint already raised for this Transaction ID")
else:
    new_data = {
    "Transaction_ID": [trans_id],
    "Issue_ID": [issue_id],
    "Transaction_Date": [trans_date],
    "Amount": [amount],
    "Receiver_UPI_ID": [receiver_UPI_ID],
    "Issue_Type": [issue_type],
    "Description": [description],
    "Time_stamp":[time_stamp]
}
complaints_df = pd.concat([complaints_df, pd.DataFrame([new_data])], ignore_index=True)
     
query = f"""I need to verify and process a refund for:
Transaction ID: {trans_id}
Date: {trans_date}
Amount: {amount}
Receiver UPI: {receiver_UPI_ID}
Description: {description}

Please verify this transaction."""    


'''
query = f"""I need to verify and process a refund for:
Transaction ID: TXN1003
Date: 2025-01-02
Amount: 2000
Receiver UPI: ATM001
Description: not sent to receiver

Please verify this transaction."""    '''
       
response = graph.invoke({'messages': [{"role": "user", "content": query}]})

for m in response['messages']:
    m.pretty_print()