from explainerdashboard.custom import *
from explainerdashboard import *


class CustomDashboard(ExplainerComponent):
    def __init__(self, explainer, name=None):
        super().__init__(explainer, title="Custom Dashboard")
        self.confusion = ConfusionMatrixComponent(explainer, name=self.name+"cm",
                            hide_selector=True, hide_percentage=True,
                            cutoff=0.75)
        self.contrib = ShapContributionsGraphComponent(explainer, name=self.name+"contrib",
                            hide_selector=True, hide_cats=True, 
                            hide_depth=True, hide_sort=True,
                            index='Rugg, Miss. Emily')
        
    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Custom Demonstration:"),
                    html.H3("How to build your own layout using ExplainerComponents.")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    self.confusion.layout(),
                ]),
                dbc.Col([
                    self.contrib.layout(),
                ])
            ])
        ])

db = ExplainerDashboard(ClassifierExplainer, CustomDashboard, hide_header=True).run()

    # def layout(self):
    #     return dbc.Container([
    #         dbc.Row([
    #             dbc.Col([
    #                 html.H3("Enter name:"),
    #                 self.index.layout()
    #             ])
    #         ]),
    #         dbc.Row([
    #             dbc.Col([
    #                 html.H3("Contributions to prediction:"),
    #                 self.contributions.layout()
    #             ]),

    #         ]),
    #         dbc.Row([

    #             dbc.Col([
    #                 html.H3("Every tree in the Random Forest:"),
    #                 self.trees.layout()
    #             ]),
    #         ])
    #     ])