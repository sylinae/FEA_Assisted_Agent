import os
import streamlit as st
from dotenv import load_dotenv

from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor
import phoenix as px

from langchain_openai import ChatOpenAI

from src.tools import (
    generate_input_file,
    run_abaqus,
    extract_von_mises_stress_from_ODB,
    extract_action,
)
from src.prompt_temp import (
    TOOL_CALLING_PROMPT_TEMPLATE,
    TOOL_UNIT_PROMPT_TEMPLATE,
    FINAL_HALLUCINATION_PROMPT_TEMPLATE,
)
from src.langgraph_agent import ToolSpec, build_langgraph_agent, build_initial_state
from phoenix.evals import (
    TOOL_CALLING_PROMPT_RAILS_MAP,
    OpenAIModel,
)
import pandas as pd

from src.eval_utils import run_eval, log_stress_eval_real_time

load_dotenv(override=True)
stress_threshold = float(os.getenv("STRESS_THRESHOLD", 360.0))

# ---------- observability (run once) ---------- #
@st.cache_resource(show_spinner=False)
def init_observability():
    tp = register(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        batch=True,
        set_global_tracer_provider=False,
    )
    LlamaIndexInstrumentor().instrument(skip_dep_check=True, tracer_provider=tp)
    LangChainInstrumentor().instrument(skip_dep_check=True, tracer_provider=tp)
    return px.launch_app()

session = init_observability()

# ---------- LLM picker ---------- #
llm_type = st.sidebar.selectbox("Select LLM type", ["gpt-4o", "gpt-4.1"])
llm_type_eval = st.sidebar.selectbox(
    "Select LLM type (judge)", ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4o", "gpt-4.1"]
)
llm_type_eval_high = st.sidebar.selectbox(
    "Select LLM type (judge high reasoning)", ["gpt-4.1"]
)

# ---------- tools ---------- #
tools = [
    ToolSpec(
        name="Abaqus_input_file_generator",
        description="Generates an Abaqus input file with an applied displacement (unit: metres). The applied displacement should not exceed 0.2 metres.",
        fn=generate_input_file,
    ),
    ToolSpec(
        name="Abaqus_job_executor",
        description="Runs an Abaqus job with `cantilever_beam.inp` and collects outputs.",
        fn=run_abaqus,
    ),
    ToolSpec(
        name="Von_Mises_stress_extractor",
        description="Extracts max Von-Mises stress from the ODB file (returns MPa).",
        fn=extract_von_mises_stress_from_ODB,
    ),
]

# ---------- init agent (once) ---------- #
if "agent_graph" not in st.session_state:
    llm = ChatOpenAI(model=llm_type)
    st.session_state.agent_graph = build_langgraph_agent(
        llm,
        tools,
    ).compile()

agent_graph = st.session_state.agent_graph

# ================== UI ================== #
st.title("Finite Element Analysis Assistant")
st.markdown("""
            This AI agent is designed to seamlessly integrate with Abaqus to automate the simulation workflow for generating a model, running the job, and extracting stress data. It streamlines the following key steps:
            1. **Model Input Generation:** Using `generate_input_file`, the agent creates an Abaqus input file (`.inp`) based on specified parameters, such as pipe thickness. The generated file is moved to a designated directory for job execution.
            2. **Job Execution:** The `run_abaqus` function initiates an Abaqus simulation using the prepared input file. Upon completion, all relevant output files are relocated to a specific directory for organization and subsequent analysis.
            3. **Stress Extraction:** Leveraging the `extract_von_mises_stress_from_ODB` function, the agent extracts Von-Mises stress data from the simulation output database (ODB). This information is saved in a file (`max_vm_stress.txt`) and stored in the same directory for easy access.
            With its modular design and reliance on tools like `os`, `subprocess`, and `shutil`, the agent ensures efficient handling of files and simulation processes, enabling robust and automated stress analysis.
            
            The current demo showcases the agent's capabilities in automating the workflow for cantilever beam simulation, focusing on stress extraction, as illustrated in the schematic figure below.
            """)

logo_file_path = "artifacts\cantilever_beam_schematic.png"
st.image(logo_file_path, width=500)

default_query = (
    "For the cantilever beam, retrieve the maximum von Mises stress when the "
    "pipe is displaced downward by 0.02 m. Then incrementally increase the "
    f"displacement until the von Mises stress reaches approximately {stress_threshold} MPa, minimising the "
    "number of simulations."
)
query = st.text_area("Enter your query:", default_query)

if st.button("Submit"):
    with st.spinner("Processing..."):
        initial_state = build_initial_state(query, tools)

        with st.expander("Show Progress"):
            client = px.Client()
            final_state = None
            last_response = None
            for state in agent_graph.stream(
                initial_state,
                {"recursion_limit": 100},
                stream_mode="values",
            ):
                final_state = state
                latest_response = state.get("latest_response")
                if latest_response and latest_response != last_response:
                    st.markdown(latest_response)
                    log_stress_eval_real_time(client)
                    last_response = latest_response

        if not final_state:
            final_state = initial_state

        final_answer = final_state.get("final_answer") or ""

        st.subheader("Final Answer:")
        st.markdown(final_answer)

        st.subheader("Intermediate Reasoning and Acting Steps:")
        with st.expander("Show the Steps"):
            for step in final_state.get("steps", []):
                for k, v in step.items():
                    st.markdown(
                        f"<span style='color:darkblue;font-weight:bold;'>{k}</span>: {v}",
                        unsafe_allow_html=True,
                    )
                st.markdown("----")
# -------------- evaluation helpers -------------- #

def tool_utilization_eval():
    judge = OpenAIModel(model=llm_type_eval_high, temperature=0)
    rails = list(TOOL_CALLING_PROMPT_RAILS_MAP.values())
    tool_definitions = [
        {"name": tool.name, "description": tool.description} for tool in tools
    ]
    return run_eval(
        span_kind="LLM",
        select=dict(
            start_time="start_time",
            question="input.value",
            output_messages="llm.output_messages",
        ),
        template=TOOL_CALLING_PROMPT_TEMPLATE,
        rails=rails,
        judge=judge,
        eval_name="Tool Utilization",
        post_process=lambda df: pd.DataFrame(
            {
                "question": df["question"],
                "tool_call": df.apply(
                    lambda r: (
                        lambda tool, args: f"{tool}({args})"
                    )(*extract_action(r.output_messages)),
                    axis=1,
                ),
                "tool_definitions": [tool_definitions] * len(df),
            },
            index=df.index,
        ),
        retries=2,
    )

def unit_eval():
    judge = OpenAIModel(model=llm_type_eval, temperature=0)
    rails = list(TOOL_CALLING_PROMPT_RAILS_MAP.values())
    tool_lookup = {tool.name: tool.description for tool in tools}
    return run_eval(
        span_kind="TOOL",
        select=dict(
            start_time="start_time",
            tool_name="tool.name",
            tool_call="input.value",
            tool_output="output.value",
        ),
        template=TOOL_UNIT_PROMPT_TEMPLATE,
        rails=rails,
        judge=judge,
        eval_name="Unit Check",
        post_process=lambda df: pd.DataFrame(
            {
                "tool_call": df["tool_call"].astype(str),
                "tool_output": df["tool_output"].astype(str),
                "tool_definition": df["tool_name"].map(tool_lookup),
            },
            index=df.index,
        ).dropna(subset=["tool_definition"]),
        retries=2,
    )

def hallucination_eval():
    judge = OpenAIModel(model=llm_type_eval, temperature=0)
    return run_eval(
        span_kind="AGENT",
        select=dict(
            start_time="start_time",
            memory="input.value",
            answer="output.value",
        ),
        template=FINAL_HALLUCINATION_PROMPT_TEMPLATE,
        rails=["hallucinated", "not"],
        judge=judge,
        eval_name="Hallucination",
        post_process=lambda df: df.tail(1),
        num_steps=1,
        retries=2,
    )

# -------------- Streamlit buttons -------------- #
st.subheader("Offline Evaluation")
if st.button("Planning Correctness: Hallucination Eval"):
    graded = hallucination_eval()
    if graded.empty:
        st.info("No agent spans found in the latest trace.")
    else:
        st.success("✅ Final answer marked correct" if graded["score"].iloc[0] == 0 else "❌ Final answer marked incorrect")
if st.button("Tool Utilization: Tool Mapping"):
    graded = tool_utilization_eval()
    if graded.empty:
        st.info("No tool‑calling LLM spans found in the latest trace.")
    else:
        st.success(f"✅ {int(graded['score'].sum())}/{len(graded)} calls marked correct")

if st.button("Tool Utilization: Unit Correctness"):
    graded = unit_eval()
    if graded.empty:
        st.info("No TOOL spans found in the latest trace.")
    else:
        st.success(f"✅ {int(graded['score'].sum())}/{len(graded)} tool calls have correct units")
