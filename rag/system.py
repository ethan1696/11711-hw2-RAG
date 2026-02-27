from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

from transformers import AutoModelForCausalLM, AutoTokenizer

from retrieval.unified_retrieval import UnifiedRetriever


RetrievalMode = Literal["dense", "sparse", "hybrid", "closed_book"]

DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_SYSTEM_PROMPT = (
    "You are a factual question-answering assistant.\n\n"
    "Use the provided context passages to answer the question.\n\n"
    "If the answer is clearly stated in the context, answer confidently.\n"
    "If the context strongly implies the answer but does not state it verbatim, you may infer cautiously.\n\n"
    "Respond with a complete sentence that directly answers the question.\n"
    "Do not mention the context in your answer.\n"
    "Do not hedge or speculate."
)
DEFAULT_USER_PROMPT_TEMPLATE = (
    "Question: {question}\n\n"
    "Context passages:\n"
    "{context_passages}\n\n"
    "Answer:"
)
DEFAULT_CONTEXT_ENTRY_TEMPLATE = (
    "[{rank}] (title: {title}; source: {url})\n"
    "{passage}"
)
DEFAULT_EMPTY_CONTEXT_TEXT = "[No retrieved context]"


@dataclass(frozen=True, slots=True)
class RetrievedDoc:
    """One retrieved context row used for generation."""

    rank: int
    chunk_uid: str
    title: str | None
    text: str
    url: str | None
    score: float
    metadata: dict[str, Any]


class RAGSystem:
    """Simple end-to-end RAG object: retrieve then generate."""

    def __init__(
        self,
        *,
        dense_embed_dir: str | Path,
        sparse_index_dir: str | Path,
        llm_model_name: str = DEFAULT_QWEN_MODEL,
        retrieval_mode: RetrievalMode = "hybrid",
        retrieval_top_k: int = 5,
        dense_device: str = "auto",
        dense_dtype: str = "auto",
        llm_torch_dtype: str | None = "auto",
        llm_device: str | None = None,
        llm_device_map: str | dict[str, Any] | None = "auto",
        dense_model_name: str = "Alibaba-NLP/gte-Qwen2-7B-instruct",
        sparse_index_name: str = "bm25_index.pkl",
        sparse_chunk_store_name: str | None = None,
        trust_remote_code: bool = True,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE,
        context_entry_template: str = DEFAULT_CONTEXT_ENTRY_TEMPLATE,
        empty_context_text: str = DEFAULT_EMPTY_CONTEXT_TEXT,
        fusion_top_k: int | None = None,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> None:
        self.retrieval_mode = retrieval_mode
        self.retrieval_top_k = int(retrieval_top_k)
        if self.retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be > 0")
        if self.retrieval_mode not in {"dense", "sparse", "hybrid", "closed_book"}:
            raise ValueError("retrieval_mode must be one of: dense, sparse, hybrid, closed_book")

        self._retriever_kwargs: dict[str, Any] = {
            "dense_embed_dir": dense_embed_dir,
            "sparse_index_dir": sparse_index_dir,
            "dense_model": dense_model_name,
            "dense_device": dense_device,
            "dense_dtype": dense_dtype,
            "sparse_index_name": sparse_index_name,
            "sparse_chunk_store_name": sparse_chunk_store_name,
            "dense_trust_remote_code": trust_remote_code,
        }
        self.retriever: UnifiedRetriever | None = None
        if self.retrieval_mode != "closed_book":
            self.retriever = UnifiedRetriever(**self._retriever_kwargs)

        self.system_prompt = str(system_prompt)
        self.user_prompt_template = str(user_prompt_template)
        self.context_entry_template = str(context_entry_template)
        self.empty_context_text = str(empty_context_text)
        self.fusion_top_k = fusion_top_k
        self.rrf_k = int(rrf_k)
        self.dense_weight = float(dense_weight)
        self.sparse_weight = float(sparse_weight)

        resolved_llm_device = None if llm_device is None else str(llm_device).strip()
        if resolved_llm_device == "":
            resolved_llm_device = None

        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if llm_torch_dtype is not None:
            model_kwargs["torch_dtype"] = llm_torch_dtype

        resolved_llm_device_map = llm_device_map
        if isinstance(resolved_llm_device_map, str):
            stripped = resolved_llm_device_map.strip()
            if not stripped:
                resolved_llm_device_map = None
            elif stripped.lower() == "auto":
                # If user specifies a concrete GPU and still wants automatic dispatch,
                # pin the whole model to that device.
                if resolved_llm_device is not None:
                    resolved_llm_device_map = {"": resolved_llm_device}
                else:
                    resolved_llm_device_map = "auto"
            elif stripped.lower() in {"none", "null"}:
                resolved_llm_device_map = None
            else:
                # Accept explicit device string as whole-model mapping.
                resolved_llm_device_map = {"": stripped}

        if resolved_llm_device_map is not None:
            model_kwargs["device_map"] = resolved_llm_device_map

        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            **model_kwargs,
        )
        if "device_map" not in model_kwargs and resolved_llm_device is not None:
            self.model.to(resolved_llm_device)
        self.model.eval()

    def _ensure_retriever(self) -> UnifiedRetriever:
        if self.retriever is None:
            self.retriever = UnifiedRetriever(**self._retriever_kwargs)
        return self.retriever

    def answer_question(
        self,
        question: str,
        *,
        top_k: int | None = None,
        mode: RetrievalMode | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        system_prompt: str | None = None,
        user_prompt_template: str | None = None,
        context_entry_template: str | None = None,
        empty_context_text: str | None = None,
        fusion_top_k: int | None = None,
        rrf_k: int | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
        return_context: bool = True,
        return_prompt: bool = False,
    ) -> dict[str, Any]:
        """Retrieve context and generate an answer with Qwen."""

        retrieval_mode = mode or self.retrieval_mode
        if retrieval_mode not in {"dense", "sparse", "hybrid", "closed_book"}:
            raise ValueError("mode must be one of: dense, sparse, hybrid, closed_book")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")

        docs: list[RetrievedDoc]
        if retrieval_mode == "closed_book":
            retrieval_k = 0
            docs = []
            context_text = ""
        else:
            retrieval_k = self.retrieval_top_k if top_k is None else int(top_k)
            if retrieval_k <= 0:
                raise ValueError("top_k must be > 0")

            raw_docs = self._ensure_retriever().search(
                question,
                top_k=retrieval_k,
                mode=retrieval_mode,
                fusion_top_k=self.fusion_top_k if fusion_top_k is None else fusion_top_k,
                rrf_k=self.rrf_k if rrf_k is None else int(rrf_k),
                dense_weight=self.dense_weight if dense_weight is None else float(dense_weight),
                sparse_weight=self.sparse_weight if sparse_weight is None else float(sparse_weight),
            )
            docs = [self._to_retrieved_doc(item) for item in raw_docs]
            context_text = self._format_context(
                docs,
                context_entry_template=(
                    self.context_entry_template
                    if context_entry_template is None
                    else str(context_entry_template)
                ),
                empty_context_text=(
                    self.empty_context_text
                    if empty_context_text is None
                    else str(empty_context_text)
                ),
            )
        resolved_user_prompt_template = (
            self.user_prompt_template
            if user_prompt_template is None
            else str(user_prompt_template)
        )
        user_prompt = self._format_template(
            resolved_user_prompt_template,
            {
                "question": question.strip(),
                "context_passages": context_text,
            },
            template_name="user_prompt_template",
        )
        resolved_system_prompt = self.system_prompt if system_prompt is None else str(system_prompt)
        messages = [
            {"role": "system", "content": resolved_system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([chat_text], return_tensors="pt").to(self.model.device)

        do_sample = temperature > 0.0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = float(temperature)

        generated_ids = self.model.generate(**model_inputs, **generation_kwargs)
        continuation_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        answer = self.tokenizer.batch_decode(
            continuation_ids,
            skip_special_tokens=True,
        )[0].strip()

        result: dict[str, Any] = {
            "question": question,
            "answer": answer,
            "mode": retrieval_mode,
            "top_k": retrieval_k,
        }
        if return_context:
            result["contexts"] = [self._doc_to_json(doc) for doc in docs]
        if return_prompt:
            result["system_prompt"] = resolved_system_prompt
            result["user_prompt"] = user_prompt
            result["model_prompt"] = chat_text
        return result

    @staticmethod
    def _to_retrieved_doc(payload: dict[str, Any]) -> RetrievedDoc:
        text = str(payload.get("text", "")).strip()
        return RetrievedDoc(
            rank=int(payload.get("rank", 0)),
            chunk_uid=str(payload.get("chunk_uid", "")).strip(),
            title=None if payload.get("title") is None else str(payload.get("title")),
            text=text,
            url=None if payload.get("url") is None else str(payload.get("url")),
            score=float(payload.get("score", 0.0)),
            metadata=dict(payload.get("metadata", {})),
        )

    @staticmethod
    def _doc_to_json(doc: RetrievedDoc) -> dict[str, Any]:
        return {
            "rank": doc.rank,
            "chunk_uid": doc.chunk_uid,
            "title": doc.title,
            "text": doc.text,
            "url": doc.url,
            "score": doc.score,
            "metadata": doc.metadata,
        }

    @classmethod
    def _format_context(
        cls,
        docs: list[RetrievedDoc],
        *,
        context_entry_template: str,
        empty_context_text: str,
    ) -> str:
        if not docs:
            return empty_context_text

        sections: list[str] = []
        for doc in docs:
            sections.append(
                cls._format_template(
                    context_entry_template,
                    {
                        "rank": doc.rank,
                        "chunk_uid": doc.chunk_uid,
                        "title": (doc.title or "Untitled").strip() or "Untitled",
                        "url": (doc.url or "unknown").strip() or "unknown",
                        "source": (doc.url or "unknown").strip() or "unknown",
                        "passage": doc.text.strip(),
                        "text": doc.text.strip(),
                        "score": doc.score,
                        "metadata": json.dumps(doc.metadata, ensure_ascii=False, sort_keys=True),
                    },
                    template_name="context_entry_template",
                )
            )
        return "\n\n".join(sections)

    @staticmethod
    def _format_template(
        template: str,
        values: dict[str, Any],
        *,
        template_name: str,
    ) -> str:
        try:
            return str(template).format(**values)
        except KeyError as exc:
            missing = str(exc).strip("'")
            raise ValueError(
                f"Missing placeholder '{missing}' required by {template_name}"
            ) from exc


__all__ = [
    "DEFAULT_CONTEXT_ENTRY_TEMPLATE",
    "DEFAULT_EMPTY_CONTEXT_TEXT",
    "DEFAULT_QWEN_MODEL",
    "RAGSystem",
    "RetrievedDoc",
    "DEFAULT_SYSTEM_PROMPT",
    "DEFAULT_USER_PROMPT_TEMPLATE",
]
