import base64
import logging
import time
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any
import requests
import asyncio

import gradio as gr  # type: ignore
from fastapi import FastAPI
from injector import inject, singleton
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
from pydantic import BaseModel

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.di import global_injector
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.chat.chat_service import ChatService, CompletionGen
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.recipes.summarize.summarize_service import SummarizeService
from private_gpt.settings.settings import settings
from private_gpt.ui.images import logo_svg

# Multi-method search pipeline – must return a dict with keys "original", "pca", "umap", "ae"
from private_gpt.search_pipeline import multi_search_pipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)
AVATAR_BOT = THIS_DIRECTORY_RELATIVE / "avatar-bot.ico"
UI_TAB_TITLE = "My Private GPT"
SOURCES_SEPARATOR = "<hr>Sources: \n"

class Modes(str, Enum):
    RAG_MODE = "RAG"
    SEARCH_MODE = "Search"
    BASIC_CHAT_MODE = "Basic"
    SUMMARIZE_MODE = "Summarize"

MODES: list[Modes] = [
    Modes.RAG_MODE,
    Modes.SEARCH_MODE,
    Modes.BASIC_CHAT_MODE,
    Modes.SUMMARIZE_MODE,
]

class Source(BaseModel):
    file: str
    page: str
    text: str

    class Config:
        frozen = True

@singleton
class PrivateGptUi:
    @inject
    def __init__(
        self,
        ingest_service: IngestService,
        chat_service: ChatService,
        chunks_service: ChunksService,
        summarizeService: SummarizeService,
    ) -> None:
        self._ingest_service = ingest_service
        self._chat_service = chat_service
        self._chunks_service = chunks_service
        self._summarize_service = summarizeService

        self._ui_block = None
        self._selected_filename = None  # (Not used in this RAG-only demo)

        # For competitor retrieval workflow:
        self._last_query = ""
        self._retrieval_results = {}

        # Initialize system prompt based on default mode
        default_mode_map = {mode.value: mode for mode in Modes}
        self._default_mode = default_mode_map.get(
            settings().ui.default_mode, Modes.RAG_MODE
        )
        self._system_prompt = self._get_default_system_prompt(self._default_mode)

        self._ingested_dataset_component = None

    # --------------------------------------------------------------------------
    # Helper: Default system prompt and mode explanation.
    # --------------------------------------------------------------------------
    @staticmethod
    def _get_default_system_prompt(mode: Modes) -> str:
        p = ""
        match mode:
            case Modes.RAG_MODE:
                p = settings().ui.default_query_system_prompt
            case Modes.BASIC_CHAT_MODE:
                p = settings().ui.default_chat_system_prompt
            case Modes.SUMMARIZE_MODE:
                p = settings().ui.default_summarization_system_prompt
            case _:
                p = ""
        return p

    @staticmethod
    def _get_default_mode_explanation(mode: Modes) -> str:
        match mode:
            case Modes.RAG_MODE:
                return "Get contextualized answers from selected files."
            case Modes.SEARCH_MODE:
                return "Find relevant chunks of text in selected files."
            case Modes.BASIC_CHAT_MODE:
                return "Chat with the LLM using its training data. Files are ignored."
            case Modes.SUMMARIZE_MODE:
                return "Generate a summary of the selected files. Prompt to customize the result."
            case _:
                return ""

    def _set_system_prompt(self, system_prompt_input: str) -> None:
        logger.info(f"Setting system prompt to: {system_prompt_input}")
        self._system_prompt = system_prompt_input

    def _set_explanatation_mode(self, explanation_mode: str) -> None:
        self._explanation_mode = explanation_mode

    def _set_current_mode(self, mode: Modes) -> Any:
        self.mode = mode
        self._set_system_prompt(self._get_default_system_prompt(mode))
        self._set_explanatation_mode(self._get_default_mode_explanation(mode))
        interactive = self._system_prompt is not None
        return [
            gr.update(placeholder=self._system_prompt, interactive=interactive),
            gr.update(value=self._explanation_mode),
        ]

    # --------------------------------------------------------------------------
    # File ingestion and management methods (unused in this demo)
    # --------------------------------------------------------------------------
    def _list_ingested_files(self) -> list[list[str]]:
        files = set()
        for ingested_document in self._ingest_service.list_ingested():
            if ingested_document.doc_metadata is None:
                continue
            file_name = ingested_document.doc_metadata.get("file_name", "[FILE NAME MISSING]")
            files.add(file_name)
        return [[row] for row in files]

    def _upload_file(self, files: list[str]) -> None:
        from pathlib import Path
        paths = [Path(file) for file in files]
        file_names = [path.name for path in paths]
        doc_ids_to_delete = []
        for ingested_document in self._ingest_service.list_ingested():
            if (
                ingested_document.doc_metadata
                and ingested_document.doc_metadata["file_name"] in file_names
            ):
                doc_ids_to_delete.append(ingested_document.doc_id)
        if doc_ids_to_delete:
            logger.info("Replacing %s existing documents.", len(doc_ids_to_delete))
            for doc_id in doc_ids_to_delete:
                self._ingest_service.delete(doc_id)
        self._ingest_service.bulk_ingest([(str(path.name), path) for path in paths])

    def _refresh_list(self) -> list[list[str]]:
        return self._list_ingested_files()

    def _delete_all_files(self) -> Any:
        ingested_files = self._ingest_service.list_ingested()
        logger.debug("Deleting %s files", len(ingested_files))
        for ingested_document in ingested_files:
            self._ingest_service.delete(ingested_document.doc_id)
        return [
            gr.List(self._list_ingested_files()),
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _delete_selected_file(self) -> Any:
        logger.debug("Deleting selected file: %s", self._selected_filename)
        for ingested_document in self._ingest_service.list_ingested():
            if ingested_document.doc_metadata and ingested_document.doc_metadata["file_name"] == self._selected_filename:
                self._ingest_service.delete(ingested_document.doc_id)
        return [
            gr.List(self._list_ingested_files()),
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _deselect_selected_file(self) -> Any:
        self._selected_filename = None
        return [
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _selected_a_file(self, select_data: gr.SelectData) -> Any:
        self._selected_filename = select_data.value
        return [
            gr.components.Button(interactive=True),
            gr.components.Button(interactive=True),
            gr.components.Textbox(self._selected_filename),
        ]

    # --------------------------------------------------------------------------
    # Helper function: yield_deltas (as in the old version)
    # --------------------------------------------------------------------------
    def yield_deltas(self, completion_gen: CompletionGen) -> Iterable[str]:
        full_response = ""
        stream = completion_gen.response
        for delta in stream:
            if isinstance(delta, str):
                full_response += str(delta)
            elif isinstance(delta, ChatResponse):
                full_response += delta.delta or ""
            yield full_response
            # time.sleep(0.02)

    # --------------------------------------------------------------------------
    # Multi-method retrieval workflow
    # --------------------------------------------------------------------------
    # def run_retrieval(self, query: str) -> tuple:
    #     """
    #     When the user clicks "Run Retrieval", call multi_search_pipeline,
    #     store the results, and return HTML strings for each competitor box.
    #     """
    #     if not query.strip():
    #         return ("<p>Please enter a valid query.</p>", "", "", "")
    #     self._last_query = query
    #     try:
    #         logger.debug("Running multi_search_pipeline with query: %s", query)
    #         results = multi_search_pipeline(query_text=query)
    #         self._retrieval_results = results
    #
    #         html_original = self._format_box("original", results.get("original"))
    #         html_pca = self._format_box("pca", results.get("pca"))
    #         html_umap = self._format_box("umap", results.get("umap"))
    #         html_ae = self._format_box("ae", results.get("ae"))
    #         return (html_original, html_pca, html_umap, html_ae)
    #     except Exception as e:
    #         logger.exception("Error during retrieval")
    #         return (f"<p>Error: {e}</p>", "", "", "")
    def run_retrieval(self, query: str) -> tuple:
        """
        When the user clicks "Run Retrieval", call multi_search_pipeline,
        store the results, and save the top-K documents for each method as files.
        """
        if not query.strip():
            return ("<p>Please enter a valid query.</p>", "", "", "")
        self._last_query = query
        try:
            logger.debug("Running multi_search_pipeline with query: %s", query)
            results = multi_search_pipeline(query_text=query)
            self._retrieval_results = results

            # Save the top-K documents for each method as files
            timestamp = int(time.time())
            for method, data in results.items():
                doc_texts = data.get("doc_texts", [])
                if not doc_texts:
                    continue
                out_path = f"generated_search_{method}_{timestamp}.txt"
                with open(out_path, "w", encoding="utf-8") as f:
                    for txt in doc_texts:
                        f.write(txt + "\n\n")
                # Upload the file to the system
                self._upload_file([out_path])

            # Return HTML strings for each method
            html_original = self._format_box("original", results.get("original"))
            html_pca = self._format_box("pca", results.get("pca"))
            html_umap = self._format_box("umap", results.get("umap"))
            html_ae = self._format_box("ae", results.get("ae"))
            return (html_original, html_pca, html_umap, html_ae)
        except Exception as e:
            logger.exception("Error during retrieval")
            return (f"<p>Error: {e}</p>", "", "", "")

    def _format_box(self, method: str, data: dict) -> str:
        """
        Format a competitor box with retrieval info.
        """
        if data is None:
            return f"<p>No data for {method.upper()}.</p>"

        retrieval_time = data.get("retrieval_time", 0)
        overlap_count = data.get("overlap_count", 0)
        doc_texts = data.get("doc_texts", [])
        top_k = len(doc_texts)
        ratio = (overlap_count / top_k) if top_k > 0 else 0

        if ratio >= 0.7:
            color = "green"
        elif ratio >= 0.3:
            color = "orange"
        else:
            color = "red"

        html = (
            f"<div style='border:1px solid #ccc; padding:10px; margin:5px;'>"
            f"<h3>{method.upper()}</h3>"
            f"<p>Retrieval Time: {retrieval_time:.2f} sec</p>"
            f"<p style='color:{color};'>Overlap: {overlap_count}/{top_k}</p>"
            f"<div style='max-height:150px; overflow-y:auto; border:1px solid #eee; padding:5px;'>"
        )
        overlap_list = data.get("overlap_list", [False] * len(doc_texts))
        for doc, is_overlap in zip(doc_texts, overlap_list):
            check = "✓" if is_overlap else ""
            snippet = doc[:100].replace("\n", " ")
            html += f"<p>{check} {snippet}...</p>"
        html += "</div></div>"
        return html

    def submit_competitor_choice(self, method: str, history: list[list[str]]) -> list[list[str]]:
        """
        When a competitor button is clicked, process the query and return the response.
        """
        if method not in self._retrieval_results:
            logger.debug("No retrieval results available for method: %s", method)
            return history + [[self._last_query, f"No retrieval results available for {method.upper()}."]]

        context_filter = None
        if self._selected_filename is not None:
            docs_ids = [
                ingested_document.doc_id
                for ingested_document in self._ingest_service.list_ingested()
                if ingested_document.doc_metadata and ingested_document.doc_metadata[
                    "file_name"] == self._selected_filename
            ]
            if docs_ids:
                context_filter = ContextFilter(docs_ids=docs_ids)

        messages = []
        if self._system_prompt:
            messages.append(ChatMessage(content=self._system_prompt, role=MessageRole.SYSTEM))
        messages.append(ChatMessage(content=self._last_query, role=MessageRole.USER))

        try:
            logger.debug("Calling chat for method: %s", method)
            response = self._chat_service.chat(
                messages=messages, use_context=True, context_filter=context_filter
            )

            # Extract the full response content
            if isinstance(response, ChatResponse):
                full_response = response.message.content
            else:
                full_response = response.response

            # Append the new interaction to the history
            return history + [[self._last_query, full_response]]

        except Exception as e:
            logger.exception("Error during LLM call for method: %s", method)
            return history + [[self._last_query, f"Error generating answer: {e}"]]

    # def submit_competitor_choice(self, method: str, history: list[list[str]]) -> Iterable[list[list[str]]]:
    #     if method not in self._retrieval_results:
    #         yield history + [[self._last_query.strip(), f"No retrieval results for {method.upper()}"]]
    #         return
    #
    #     context_filter = None
    #     if self._selected_filename:
    #         docs_ids = [
    #             doc.doc_id for doc in self._ingest_service.list_ingested()
    #             if doc.doc_metadata and doc.doc_metadata["file_name"] == self._selected_filename
    #         ]
    #         if docs_ids:
    #             context_filter = ContextFilter(docs_ids=docs_ids)
    #
    #     messages = []
    #     if self._system_prompt:
    #         messages.append(ChatMessage(content=self._system_prompt, role=MessageRole.SYSTEM))
    #     messages.append(ChatMessage(content=self._last_query, role=MessageRole.USER))
    #
    #     stream = self._chat_service.stream_chat(messages=messages, use_context=True, context_filter=context_filter)
    #
    #     assistant_response = ""
    #     for delta in self.yield_deltas(stream):
    #         assistant_response = delta.strip()
    #         if assistant_response:  # only yield non-empty responses
    #             yield history + [[self._last_query.strip(), assistant_response]]
    #
    #     if hasattr(stream, "sources") and stream.sources:
    #         formatted_sources = self._format_sources(stream.sources)
    #         assistant_response += f"\n\n**Sources:**\n{formatted_sources}"
    #         yield history + [[self._last_query.strip(), assistant_response]]
    #
    # def _format_sources(self, sources: list[Chunk]) -> str:
    #     """
    #     Format sources information into a string.
    #     """
    #     formatted_sources = ""
    #     for i, source in enumerate(sources, start=1):
    #         file_name = source.document.doc_metadata.get("file_name", "Unknown File")
    #         page_label = source.document.doc_metadata.get("page_label", "Unknown Page")
    #         text_snippet = source.text[:100]  # Show a snippet of the source text
    #         formatted_sources += f"{i}. **{file_name}** (Page {page_label}): {text_snippet}...\n"
    #     return formatted_sources
    # --------------------------------------------------------------------------
    # Build the UI (ChatInterface removed)
    # --------------------------------------------------------------------------
    def _build_ui_blocks(self) -> gr.Blocks:
        logger.debug("Creating the UI blocks")
        with gr.Blocks(
                title=UI_TAB_TITLE,
                theme=gr.themes.Soft(primary_hue="slate"),
                css="""
                .logo { 
                    display: flex;
                    background-color: #C7BAFF;
                    height: 80px;
                    border-radius: 8px;
                    align-items: center;
                    justify-content: center;
                }
                .logo img { height: 25% }
                .avatar-image { background-color: antiquewhite; border-radius: 2px; }
            """
        ) as blocks:
            with gr.Row():
                gr.HTML(f"<div class='logo'>DeepSearch</div>")
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    default_mode = self._default_mode
                    mode = gr.Radio(
                        [m.value for m in MODES],
                        label="Mode",
                        value=default_mode,
                    )
                    explanation_mode = gr.Textbox(
                        placeholder=self._get_default_mode_explanation(default_mode),
                        show_label=False,
                        max_lines=3,
                        interactive=False,
                    )
                    upload_button = gr.components.UploadButton(
                        "Upload File(s)",
                        type="filepath",
                        file_count="multiple",
                        size="sm",
                    )

                    self._ingested_dataset_component = gr.List(
                        self._list_ingested_files,
                        headers=["File name"],
                        label="Ingested Files",
                        height=235,
                        interactive=False,
                    )
                    upload_button.upload(
                        self._upload_file,
                        inputs=upload_button,
                        outputs=self._ingested_dataset_component,
                    )

                    self._ingested_dataset_component.change(
                        self._list_ingested_files,
                        outputs=self._ingested_dataset_component,
                    )
                    refresh_button = gr.Button("Refresh Files", size="sm")
                    refresh_button.click(
                        fn=self._refresh_list,
                        inputs=[],
                        outputs=self._ingested_dataset_component
                    )

                    deselect_file_button = gr.Button("De-select selected file", size="sm", interactive=False)
                    selected_text = gr.Textbox("All files", label="Selected for Query or Deletion", max_lines=1)
                    delete_file_button = gr.Button(
                        "🗑️ Delete selected file",
                        size="sm",
                        visible=settings().ui.delete_file_button_enabled,
                        interactive=False,
                    )
                    delete_files_button = gr.Button(
                        "⚠️ Delete ALL files",
                        size="sm",
                        visible=settings().ui.delete_all_files_button_enabled,
                    )
                    deselect_file_button.click(
                        self._deselect_selected_file,
                        outputs=[delete_file_button, deselect_file_button, selected_text],
                    )
                    self._ingested_dataset_component.select(
                        fn=self._selected_a_file,
                        outputs=[delete_file_button, deselect_file_button, selected_text],
                    )
                    delete_file_button.click(
                        self._delete_selected_file,
                        outputs=[self._ingested_dataset_component, delete_file_button, deselect_file_button,
                                 selected_text],
                    )
                    delete_files_button.click(
                        self._delete_all_files,
                        outputs=[self._ingested_dataset_component, delete_file_button, deselect_file_button,
                                 selected_text],
                    )
                    system_prompt_input = gr.Textbox(
                        placeholder=self._system_prompt,
                        label="System Prompt",
                        lines=2,
                        interactive=True,
                        render=False,
                    )
                    mode.change(
                        self._set_current_mode,
                        inputs=mode,
                        outputs=[system_prompt_input, explanation_mode],
                    )
                    system_prompt_input.blur(
                        self._set_system_prompt,
                        inputs=system_prompt_input,
                    )
                with gr.Column(scale=7):
                    with gr.Row():
                        query_input = gr.Textbox(label="Enter Query", placeholder="Type your query here", lines=1, scale=5)
                        run_retrieval_btn = gr.Button("Run Retrieval", variant="primary", scale=1)
                    with gr.Row():
                        html_original = gr.HTML(label="Original")
                        html_pca = gr.HTML(label="PCA")
                    with gr.Row():
                        html_umap = gr.HTML(label="UMAP")
                        html_ae = gr.HTML(label="AE")
                    with gr.Row():
                        btn_original = gr.Button("Submit Original")
                        btn_pca = gr.Button("Submit PCA")
                        btn_umap = gr.Button("Submit UMAP")
                        btn_ae = gr.Button("Submit AE")
                    # Use a Chatbot for displaying LLM answers with streaming updates
                    competitor_answer = gr.Chatbot(label="LLM Answer", elem_classes=["chatbot"])
                    run_retrieval_btn.click(
                        fn=self.run_retrieval,
                        inputs=query_input,
                        outputs=[html_original, html_pca, html_umap, html_ae],
                    )
                    btn_original.click(
                        fn=lambda *args: self.submit_competitor_choice("original", history=competitor_answer.value),
                        inputs=[],
                        outputs=[competitor_answer],
                    )
                    btn_pca.click(
                        fn=lambda *args: self.submit_competitor_choice("pca", history=competitor_answer.value),
                        inputs=[],
                        outputs=[competitor_answer],
                    )
                    btn_umap.click(
                        fn=lambda *args: self.submit_competitor_choice("umap", history=competitor_answer.value),
                        inputs=[],
                        outputs=[competitor_answer],
                    )
                    btn_ae.click(
                        fn=lambda *args: self.submit_competitor_choice("ae", history=competitor_answer.value),
                        inputs=[],
                        outputs=[competitor_answer],
                    )

        return blocks
    # def _build_ui_blocks(self) -> gr.Blocks:
    #     logger.debug("Creating the UI blocks")
    #     with gr.Blocks(
    #         title=UI_TAB_TITLE,
    #         theme=gr.themes.Soft(primary_hue="slate"),
    #         css="""
    #             .logo {
    #                 display: flex;
    #                 background-color: #C7BAFF;
    #                 height: 80px;
    #                 border-radius: 8px;
    #                 align-items: center;
    #                 justify-content: center;
    #             }
    #             .logo img { height: 25% }
    #             .avatar-image { background-color: antiquewhite; border-radius: 2px; }
    #         """
    #     ) as blocks:
    #         with gr.Row():
    #             gr.HTML(f"<div class='logo'>DeepSearch</div>")
    #         with gr.Row(equal_height=False):
    #             with gr.Column(scale=3):
    #                 default_mode = self._default_mode
    #                 mode = gr.Radio(
    #                     [m.value for m in MODES],
    #                     label="Mode",
    #                     value=default_mode,
    #                 )
    #                 explanation_mode = gr.Textbox(
    #                     placeholder=self._get_default_mode_explanation(default_mode),
    #                     show_label=False,
    #                     max_lines=3,
    #                     interactive=False,
    #                 )
    #                 upload_button = gr.components.UploadButton(
    #                     "Upload File(s)",
    #                     type="filepath",
    #                     file_count="multiple",
    #                     size="sm",
    #                 )
    #
    #                 self._ingested_dataset_component = gr.List(
    #                     self._list_ingested_files,
    #                     headers=["File name"],
    #                     label="Ingested Files",
    #                     height=235,
    #                     interactive=False,
    #                 )
    #                 upload_button.upload(
    #                     self._upload_file,
    #                     inputs=upload_button,
    #                     outputs=self._ingested_dataset_component,
    #                 )
    #
    #                 self._ingested_dataset_component.change(
    #                     self._list_ingested_files,
    #                     outputs=self._ingested_dataset_component,
    #                 )
    #                 refresh_button = gr.Button("Refresh Files", size="sm")
    #                 refresh_button.click(
    #                     fn=self._refresh_list,
    #                     inputs=[],
    #                     outputs=self._ingested_dataset_component
    #                 )
    #
    #                 deselect_file_button = gr.Button("De-select selected file", size="sm", interactive=False)
    #                 selected_text = gr.Textbox("All files", label="Selected for Query or Deletion", max_lines=1)
    #                 delete_file_button = gr.Button(
    #                     "🗑️ Delete selected file",
    #                     size="sm",
    #                     visible=settings().ui.delete_file_button_enabled,
    #                     interactive=False,
    #                 )
    #                 delete_files_button = gr.Button(
    #                     "⚠️ Delete ALL files",
    #                     size="sm",
    #                     visible=settings().ui.delete_all_files_button_enabled,
    #                 )
    #                 deselect_file_button.click(
    #                     self._deselect_selected_file,
    #                     outputs=[delete_file_button, deselect_file_button, selected_text],
    #                 )
    #                 self._ingested_dataset_component.select(
    #                     fn=self._selected_a_file,
    #                     outputs=[delete_file_button, deselect_file_button, selected_text],
    #                 )
    #                 delete_file_button.click(
    #                     self._delete_selected_file,
    #                     outputs=[self._ingested_dataset_component, delete_file_button, deselect_file_button, selected_text],
    #                 )
    #                 delete_files_button.click(
    #                     self._delete_all_files,
    #                     outputs=[self._ingested_dataset_component, delete_file_button, deselect_file_button, selected_text],
    #                 )
    #                 system_prompt_input = gr.Textbox(
    #                     placeholder=self._system_prompt,
    #                     label="System Prompt",
    #                     lines=2,
    #                     interactive=True,
    #                     render=False,
    #                 )
    #                 mode.change(
    #                     self._set_current_mode,
    #                     inputs=mode,
    #                     outputs=[system_prompt_input, explanation_mode],
    #                 )
    #                 system_prompt_input.blur(
    #                     self._set_system_prompt,
    #                     inputs=system_prompt_input,
    #                 )
    #             with gr.Column(scale=7):
    #                 with gr.Row():
    #                     query_input = gr.Textbox(label="Enter Query", placeholder="Type your query here", lines=1)
    #                     run_retrieval_btn = gr.Button("Run Retrieval", variant="primary")
    #                 with gr.Row():
    #                     html_original = gr.HTML(label="Original")
    #                     html_pca = gr.HTML(label="PCA")
    #                 with gr.Row():
    #                     html_umap = gr.HTML(label="UMAP")
    #                     html_ae = gr.HTML(label="AE")
    #                 with gr.Row():
    #                     btn_original = gr.Button("Submit Original")
    #                     btn_pca = gr.Button("Submit PCA")
    #                     btn_umap = gr.Button("Submit UMAP")
    #                     btn_ae = gr.Button("Submit AE")
    #                 # Use a Chatbot for displaying LLM answers with streaming updates
    #                 competitor_answer = gr.Chatbot(label="LLM Answer", elem_classes=["chatbot"])
    #                 run_retrieval_btn.click(
    #                     fn=self.run_retrieval,
    #                     inputs=query_input,
    #                     outputs=[html_original, html_pca, html_umap, html_ae],
    #                 )
    #                 btn_original.click(
    #                     fn=lambda *args: self.submit_competitor_choice("original", history=competitor_answer.value),
    #                     inputs=[],
    #                     outputs=[competitor_answer],
    #                 )
    #                 btn_pca.click(
    #                     fn=lambda *args: self.submit_competitor_choice("pca", history=competitor_answer.value),
    #                     inputs=[],
    #                     outputs=[competitor_answer],
    #                 )
    #                 btn_umap.click(
    #                     fn=lambda *args: self.submit_competitor_choice("umap", history=competitor_answer.value),
    #                     inputs=[],
    #                     outputs=[competitor_answer],
    #                 )
    #                 btn_ae.click(
    #                     fn=lambda *args: self.submit_competitor_choice("ae", history=competitor_answer.value),
    #                     inputs=[],
    #                     outputs=[competitor_answer],
    #                 )
    #
    #     return blocks

    def get_ui_blocks(self) -> gr.Blocks:
        if self._ui_block is None:
            self._ui_block = self._build_ui_blocks()
        return self._ui_block

    def mount_in_app(self, app: FastAPI, path: str) -> None:
        blocks = self.get_ui_blocks()
        blocks.queue()
        logger.info("Mounting the gradio UI at path=%s", path)
        gr.mount_gradio_app(app, blocks, path=path, favicon_path=AVATAR_BOT)

if __name__ == "__main__":
    from private_gpt.di import global_injector
    ui = global_injector.get(PrivateGptUi)
    _blocks = ui.get_ui_blocks()
    _blocks.queue()
    _blocks.launch(debug=False, show_api=False)
