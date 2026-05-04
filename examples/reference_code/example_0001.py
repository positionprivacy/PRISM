from manim import *


class ExampleLinearFunction(Scene):
    def construct(self):
        title = Text("Linear Function", font_size=36).to_edge(UP)
        axes = Axes(x_range=[-3, 3], y_range=[-2, 4], x_length=6, y_length=4).shift(DOWN * 0.3)
        graph = axes.plot(lambda x: 0.8 * x + 1, color=BLUE)
        formula = MathTex("y=ax+b", font_size=34).next_to(title, DOWN)
        slope = Text("slope a", font_size=24, color=YELLOW).next_to(graph, RIGHT)
        intercept = Text("intercept b", font_size=24, color=GREEN).next_to(axes.c2p(0, 1), LEFT)

        self.play(Write(title), Write(formula))
        self.play(Create(axes))
        self.play(Create(graph))
        self.play(FadeIn(slope), FadeIn(intercept))
        self.wait(1)
