import sys
import logging
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# Ensure local directories can be imported
sys.path.append(".")

from nodes import AgentState, fetch_weather, judge_conditions, send_notification

# Set up logging format and levels
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("surf_agent")

# 1. Initialize and Build the LangGraph Workflow
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("fetch", fetch_weather)
workflow.add_node("judge", judge_conditions)
workflow.add_node("notify", send_notification)

# Add Edges
workflow.set_entry_point("fetch")
workflow.add_edge("fetch", "judge")
workflow.add_conditional_edges(
    "judge",
    lambda state: "notify" if state.get("decision") == "GO" else END
)
workflow.add_edge("notify", END)

# Compile the graph
app = workflow.compile()

def run_agent():
    """
    Executes the compiled Surf-Predictor-Agent graph.
    """
    logger.info("Initializing Surf-Predictor-Agent execution...")
    # Load .env file variables
    load_dotenv()
    
    initial_state = {
        "forecast_data": None,
        "decision": None,
        "reasoning": None
    }
    
    try:
        final_state = app.invoke(initial_state)
        logger.info("Surf-Predictor-Agent completed execution successfully.")
        print("\n--- FINAL AGENT STATE ---")
        print(f"Decision:  {final_state.get('decision')}")
        print(f"Reasoning: {final_state.get('reasoning')}")
        print("-------------------------\n")
        return final_state
    except Exception as e:
        logger.critical(f"Agent execution encountered an unhandled exception: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_agent()
