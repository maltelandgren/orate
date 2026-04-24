"""Smoke test: render the LLMProtagonist alone."""
from manim import Scene, config, FadeOut
import theme
from theme import Paper
from llm import LLMProtagonist, LogitItem


class SmokeLLM(Scene):
    def construct(self):
        self.camera.background_color = Paper.bg
        bg = theme.paper_grid(step=0.5, opacity=0.35)
        self.add(bg)

        llm = LLMProtagonist(palette="paper", label="LLM")
        llm.shift([0, -0.5, 0])
        self.play(*[
            m.animate.set_opacity(m.get_fill_opacity() or 1.0)
            for m in [llm]
        ], run_time=0.01)
        self.add(llm)

        llm.stream_tokens(self, ["the ", "cat ", "sat ", "on ", "the "],
                          speed=0.15)
        llm.pulse_thinking(self, duration=0.7)
        llm.stream_tokens(self, ["mat", "."], speed=0.15)
        self.wait(0.3)

        logits = [
            LogitItem("mat", 0.42),
            LogitItem("couch", 0.21),
            LogitItem("42", 0.18),
            LogitItem("moon", 0.10),
            LogitItem("!!", 0.09),
        ]
        llm.clear_output(self)
        llm.stream_tokens(self, ["the ", "cat ", "sat ", "on ", "the "],
                          speed=0.1)
        llm.open_logits(self, logits)
        self.wait(0.3)
        llm.apply_grammar_mask(self, mask_indices=[1, 2, 3, 4])
        self.wait(0.3)
        llm.choose_logit(self, 0)
        self.wait(0.5)
        llm.close_logits(self)
        self.wait(0.4)
