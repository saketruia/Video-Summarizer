import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai
import time
import os
from pathlib import Path
import tempfile
from dotenv import load_dotenv


load_dotenv()

API_KEY=os.getenv("GOOGLE_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)

st.set_page_config(
    page_title="Video Summarizer Agent",
    page_icon=":video_camera:",
    layout="wide",
)


st.title("Video Summarizer Agent")
st.header("This App summarizes your video")


def initialize_agent():
    return Agent(
        name="Video Summarizer Agent",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True
    )


multimodal_Agent = initialize_agent()

video_file= st.file_uploader(
    "Upload a video file", type=["mp4", "mov", "avi"], help="Upload a video for AI Analysis"
)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path=temp_video.name

    st.video(video_path, format="video/mp4", start_time=0)

    user_query=st.text_area(
        "What insights are you seeking from the video",
        placeholder="Ask Anything about the video content",
        help="Provide specific questions"
    )

    if st.button("Analyze Video", key="analyze_video_button"):
        if not user_query:
            st.warning("please enter a question or insight to analyze the video.")
        else:
            try:
                with st.spinner("Analyzing Video..."):
                    processed_video=upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video= get_file(processed_video.name)

                    analysis_prompt=(
                        f"""
                        Analyze the uploaded video for content and context.
                        Respond to the following querry using video insights and supplementary web research:
                        {user_query}

                        Provide a detailed, user friendly and actionable response.
                        """
                    )

                    response =multimodal_Agent.run(analysis_prompt, videos=[processed_video])

                st.subheader("Analysis Result")
                st.markdown(response.content)

            except Exception as error:
                st.error(f"Error occurred during video analysis:  {error}")
            finally:
                Path(video_path).unlink(missing_ok=True)
            
    else:
        st.info("Upload a video file to begin analysis")


st.markdown(
    """
    <style>
    .stTextArea textare{
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


