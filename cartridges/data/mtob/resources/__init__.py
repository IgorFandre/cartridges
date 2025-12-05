import json
from pathlib import Path
from typing import Literal

from capsules.generate.generate_training import BaseSectionedContextConfig, BaseContextConfig
from capsules.generate.structs import Context, Section, SectionedContext
from capsules.generate.utils import unopionated_section_maker


dataset_root = Path(__file__).resolve().parent


def load_book_long():
    return (dataset_root / "grammar_book_for_claude_long.txt").read_text()


def load_book_medium():
    return (dataset_root / "grammar_book_for_claude_medium.txt").read_text()


def load_book_full():
    return (dataset_root / "grammar_book.txt").read_text()


def load_book_full_tex():
    return (dataset_root / "grammar_book.tex").read_text()


def load_wordlist():
    return json.loads((dataset_root / "wordlist.json").read_text())


def load_test_ek():
    data = json.loads((dataset_root / "test_examples_ek.json").read_text())[1:]
    assert len(data) == 50
    return data


def load_test_ke():
    data = json.loads((dataset_root / "test_examples_ke.json").read_text())[1:]
    assert len(data) == 50
    return data


def load_train_examples():
    data = json.loads((dataset_root / "train_examples.json").read_text())[1:]
    return data


def wordlist_to_lines(wordlist: dict[str, list[str] | str]) -> list:
    return (
        [
            f"{source}: {','.join(target) if isinstance(target, list) else target}"
            for (source, target) in wordlist.items()
        ]
    )


class KalamangContextConfig(BaseContextConfig):
    book_type: Literal["long", "medium", "full", "full_tex"] = "long"

    def instantiate(self) -> Context:
        if self.book_type == "long":
            book = load_book_long()
        elif self.book_type == "medium":
            book = load_book_medium()
        elif self.book_type == "full":
            book = load_book_full()
        elif self.book_type == "full_tex":
            book = load_book_full_tex()

        wordlist = load_wordlist()

        kalamang_to_english_wordlist = wordlist_to_lines(wordlist["ke"])
        english_to_kalamang_wordlist = wordlist_to_lines(wordlist["ek"])

        parallel_sentences = (
            [
                f"{row["original"]}: {row["translation"]}"
                for row in load_train_examples()
            ]
        )

        sections = [
            Section(
                content=book,
                desc="Kalamang to English Grammar Book",
                path="mtob/kalamang_to_english_grammar_book",
            ),
            Section(
                content="\n".join(kalamang_to_english_wordlist),
                desc="Kalamang to English Wordlist",
                path="mtob/kalamang_to_english_wordlist",
            ),
            Section(
                content="\n".join(english_to_kalamang_wordlist),
                desc="English to Kalamang Wordlist",
                path="mtob/english_to_kalamang_wordlist",
            ),
            Section(
                content="\n".join(parallel_sentences),
                desc="Parallel Sentences",
                path="mtob/parallel_sentences",
            )
        ]

        return Context(
            title="Kalamang Manual",
            sections=sections,
        )

class KalamangSectionedConfig(BaseSectionedContextConfig):

    max_tokens_per_section: int
    book_size: Literal["long" , "medium" , "full"] = "long"

    def instantiate(self, tokenizer) -> SectionedContext:
        if self.book_size == "long":
            book_content = load_book_long().splitlines()
        elif self.book_size == "medium":
            book_content = load_book_medium().splitlines()
        elif self.book_size == "full":
            book_content = load_book_full().splitlines()
        else:
            raise ValueError(f"Unknown book size: {self.book_size}")

        wordlist = load_wordlist()

        kalamang_to_english_wordlist = wordlist_to_lines(wordlist["ke"])
        english_to_kalamang_wordlist = wordlist_to_lines(wordlist["ek"])

        parallel_sentences = (
            [
                f"{row["original"]}: {row["translation"]}"
                for row in load_train_examples()
            ]
        )

        sections = [
            *unopionated_section_maker(
                book_content,
                "Kalamang to English Grammar Book",
                self.max_tokens_per_section,
                tokenizer,
            ),
            *unopionated_section_maker(
                kalamang_to_english_wordlist,
                "Kalamang to English Wordlist",
                self.max_tokens_per_section,
                tokenizer,
            ),
            *unopionated_section_maker(
                english_to_kalamang_wordlist,
                "English to Kalamang Wordlist",
                self.max_tokens_per_section,
                tokenizer,
            ),
            *unopionated_section_maker(
                parallel_sentences,
                "Parallel Sentences",
                self.max_tokens_per_section,
                tokenizer,
            ),
        ]

        return SectionedContext(
            sections=sections,
            title="Kalamang to English translation manuals",
        )


class KalamangLaTeXSectionedConfig(BaseSectionedContextConfig):

    max_tokens_per_section: int

    def instantiate(self, tokenizer) -> SectionedContext:
        book_long = load_book_full_tex().splitlines()

        parallel_sentences = (
            [
                f"{row["original"]}: {row["translation"]}"
                for row in load_train_examples()
            ]
        )

        sections = [
            *unopionated_section_maker(
                book_long,
                "Kalamang to English Grammar Book",
                self.max_tokens_per_section,
                tokenizer,
            ),
            *unopionated_section_maker(
                parallel_sentences,
                "Parallel Sentences",
                self.max_tokens_per_section,
                tokenizer,
            ),
        ]

        return SectionedContext(
            sections=sections,
            title="Kalamang to English translation manuals",
        )



