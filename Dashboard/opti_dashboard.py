import os

import streamlit as st


def main() -> None:

    # app configs
    st.set_page_config(layout="wide")  # use all the available space

    st.markdown(
        """
        # TTP Optimisation Dashboard


        #### 👈 Select a tab from the sidebar

    """
    )
    # scenarios = ["standard outputs"] + sorted(
    #     [
    #         x
    #         for x in os.listdir("data/scenarios")
    #         if os.path.isdir(os.path.join("data", "scenarios", x))
    #     ]
    # )
    #
    # def refresh_data() -> None:
    #     for key in st.session_state.keys():
    #         if key not in ["output_folder", "chosen_scenario"]:
    #             del st.session_state[key]
    #
    # chosen_scenario = st.session_state.get("chosen_scenario")
    # if chosen_scenario:
    #     scenarios = [chosen_scenario] + [x for x in scenarios if x != chosen_scenario]
    # chosen_scenario = st.selectbox(
    #     "scenario", options=scenarios, on_change=refresh_data
    # )
    # if chosen_scenario:
    #     st.session_state["chosen_scenario"] = chosen_scenario
    #     st.session_state["input_folder"] = (
    #         os.path.join("data", "scenarios", chosen_scenario, "inputs")
    #         if chosen_scenario != "standard outputs"
    #         else os.path.join("data", "inputs")
    #     )
    #     st.session_state["data_model_folder"] = (
    #         os.path.join("data", "scenarios", chosen_scenario, "data_model")
    #         if chosen_scenario != "standard outputs"
    #         else os.path.join("data", "data_model")
    #     )
    #     st.session_state["output_folder"] = (
    #         os.path.join("data", "scenarios", chosen_scenario, "outputs")
    #         if chosen_scenario != "standard outputs"
    #         else os.path.join("data", "outputs")
    #     )
    #     st.session_state["src_folder"] = "src"


if __name__ == "__main__":
    main()