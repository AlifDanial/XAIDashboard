from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression

from explainerdashboard import *
from explainerdashboard.datasets import *
from explainerdashboard.custom import *

import dash_html_components as html
import dash_bootstrap_components as dbc

from explainerdashboard.custom import *

from explainerdashboard import ExplainerDashboard
import pandas as pd
import csv
from pathlib import Path

projecttitle = ""
auto=0
algo = [RandomForestRegressor]

#import data
train_csv = Path(__file__).resolve().parent / 'StudentPerformance_Prediction_Regression.csv'
df = pd.read_csv(train_csv)
has_header = csv.Sniffer().has_header(open(train_csv).read(2048))

#column variables
IDColumn = "age"
predict = "Final_Score"
drop = ['address','Fedu','Mjob']

#id column set 
df.set_index(IDColumn, drop=True, inplace=True)
df.index.name = IDColumn

#predict to columns
result = predict.replace(' ', '_')

#convert list drop
converter = lambda x: x.replace(' ', '_')
drop = list(map(converter, drop))
drop

#space to underscore for all headers
if has_header == False:
  df.columns = ['co_' + str(i+1) for i in range(len(df.iloc[0].values))]
df.columns = df.columns.str.replace(' ','_')

#drop unused columns
df.drop(columns=drop, axis=1, inplace=True)

cols = df.columns
cols_dtypes = df.dtypes
is_null = df.isnull().any()
null_cols = []
for col in cols:
  if is_null[col] == True:
    null_cols.append(col)
    if cols_dtypes[col] == 'float':
      df[col].fillna(df[col].mean(), inplace = True)
    else:
      df[col].fillna(df[col].mode()[0], inplace = True)
df.drop_duplicates(inplace = True)

catCols = df.select_dtypes("object").columns
nCols = df.select_dtypes(exclude ='object').columns
catCols= list(set(catCols))
nCols= list(set(nCols))

catCols = df.select_dtypes("object").columns
catCols= list(set(catCols))
df = pd.get_dummies(df, columns=catCols)

df1 = df[[result]]
df.drop([result], axis=1, inplace=True)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df, df1, test_size=0.3, random_state=42, shuffle=True)

if auto == 1:
    b = []
    a = [RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, ExtraTreesRegressor]

    for i in a:
            model = i().fit(x_train, y_train.values.ravel())
  
            #testing training accuracy
            from sklearn.metrics import mean_squared_error
            from sklearn.metrics import explained_variance_score
            from sklearn.metrics import max_error
            from sklearn.metrics import mean_absolute_error
            from sklearn.metrics import mean_squared_error
            from sklearn.metrics import r2_score

            y_pred = model.predict(x_test)
            x = model.score(x_test,y_pred) 
            y = mean_squared_error(y_test, y_pred)
            z = explained_variance_score(y_test, y_pred)
            z1 = max_error(y_test, y_pred)
            z2 = mean_absolute_error(y_test, y_pred)
            # z3 = mean_squared_error(y_test, y_pred)
            z4 = r2_score(y_test, y_pred)
            # b.append(x - y + z - z1 - z2 + z4)
            v = z1+y+z2
            b.append((300 - (v))/300 / 0.01)
            print("Accuracy:", x ,"MSE:", y ,"EVS:", z ,"ME:", z1 ,"MAE:", z2 ,"R2:", z4)
            
    
    print(b)
    init_score = min(b)
    index = b.index(init_score)

    def switch(index):
        if index == 0:
            return [RandomForestRegressor]
        elif index == 1:
            return [GradientBoostingRegressor]
        elif index == 2:
            return [BaggingRegressor]
        elif index == 3:
            return [ExtraTreesRegressor]

    print(switch(index))
    automodel = switch(index)

    from sklearn import preprocessing
    import numpy as np
    b = preprocessing.normalize([b])
    smallest = np.min(b)
    #save best score as final best score in home page
    best_score=((1 - smallest) / 0.01)
    print(smallest)
    print(best_score)

    for i in automodel:
        model = i().fit(x_train, y_train.values.ravel())
        explainer = RegressionExplainer(model, x_test, y_test,                                 
                                cats=catCols,  
                                descriptions={},
                                units="$", shap_kwargs=dict(check_additivity=False))

        explainer.plot_contributions(0)

        class CustomDashboard(ExplainerComponent):
            def __init__(self, explainer, name=None, **kwargs):
                super().__init__(explainer, title="Impact")
                self.shap_summary = ShapSummaryComponent(explainer, name=self.name+"summary",
                                        hide_title=True, hide_depth=True)
                self.precision = PrecisionComponent(explainer, name=self.name+"precision",
                                        hide_cutoff=True, hide_binsize=True, 
                                        hide_binmethod=True, hide_multiclass=True,
                                        hide_selector=True, hide_title=True, hide_footer=True,
                                        cutoff=None, hide_depth=True)
                
                self.featuredesc = FeatureDescriptionsComponent(explainer, name=self.name+"featuredesc",hide_title=True, hide_subtitle=True)
                # self.connector = ShapSummaryDependenceConnector(self.shap_summary, self.shap_dependence)
                self.predictiontab = FeatureInputComponent(explainer, name=self.name+"predictiontab",title="What If..", subtitle="Adjust the column values to change the prediction", hide_index=False, hide_depth=False,fill_row_first=True)
                self.predictiongraph = ShapContributionsGraphComponent(explainer, name=self.name+"predictiongraph",hide_title=False, subtitle="How has each value contributed to the prediction?", hide_index=False, hide_depth=False, hide_sort=False, feature_input_component=self.predictiontab, hide_selector=False)
                self.predictioncontrib = ShapContributionsTableComponent(explainer, name=self.name+"predictioncontrib",hide_title=False, subtitle="How has each value contributed to the prediction?", hide_index=False, hide_depth=False, hide_sort=False, feature_input_component=self.predictiontab, hide_selector=False )
                self.predictionsum = ClassifierPredictionSummaryComponent(explainer, name=self.name+"predictionsum", **kwargs)
                self.predictionsum1 = ImportancesComponent(explainer, name=self.name+"predictionsum1", hide_type=True, hide_selector=True, hide_title=True, hide_subtitle=True)
                
                
            def layout(self):
                return dbc.Container([     
                    # dbc.Row([
                    #     dbc.Col([
                    #         html.H3("Feature Descriptions"),
                    #         self.featuredesc.layout()
                    #     ], style=dict(margin=20)),
                    #     ], style=dict(margin=20)),     
                    dbc.Row([
                        dbc.Col([
                            html.H3("Column Impact"),
                            html.Div("Analyze the impact each column has sorted from highest to lowest on the prediction."),
                            html.Div(f"{self.explainer.columns_ranked_by_shap()[0]} had the biggest impact"
                                    f", followed by {self.explainer.columns_ranked_by_shap()[1]}"
                                    f" and {self.explainer.columns_ranked_by_shap()[2]}."),                                                            
                        ], style=dict(margin=10)), 
                        ], style=dict(margin=10)),
                    dbc.Row([
                        dbc.Col([                    
                            self.predictionsum1.layout()
                        ], style=dict(margin=10)),
                        ], style=dict(margin=10)),        
                    dbc.Row([
                        dbc.Col([
                            html.H3("Individual Value Impact"),
                            html.Div("Explore the values from each column that have the greatest and least impact on the prediction."),
                            # self.predictiontab.layout()
                        ], style=dict(margin=10)),
                        ], style=dict(margin=10)),        
                    dbc.Row([
                        dbc.Col([                
                            self.predictiontab.layout()
                        ], style=dict(margin=10)),
                        ], style=dict(margin=10)),        
                    # dbc.Row([
                    #     dbc.Col([                    
                    #         self.predictiongraph.layout()
                    #     ], width=4, style=dict(margin=20)),
                    #     ]),
                    dbc.Row([
                        dbc.Col(self.predictioncontrib.layout(),style=dict(margin=10)),
                        dbc.Col(self.predictiongraph.layout(),style=dict(margin=10)),
                        # dbc.Col([                    
                        #     self.predictiongraph.layout()
                        # ]),
                        ],style=dict(margin=10)),
                    # dbc.Row([
                    #     dbc.Col([
                    #         html.H3("Prediction Summary"),
                    #         self.predsum.layout()
                    #     ], style=dict(margin=20)),
                    #     ]),
                ])

        class CustomPredictionsTab(ExplainerComponent):
            def __init__(self, explainer, name=None):
                super().__init__(explainer, title="Impact Relationship")
                self.shap_summary = ShapSummaryComponent(explainer, name=self.name+"summary",
                                    hide_title=True, hide_subtitle=True)
                self.shap_dependence = ShapDependenceComponent(explainer, name=self.name+"dependence", title="Impact Relationship", subtitle="Relationship between Feature value and Impact value")
                self.connector = ShapSummaryDependenceConnector(self.shap_summary, self.shap_dependence)                                

            def layout(self):
                return dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            html.H3("Link between Columns & Impact"),
                            html.Div("Analyze the relationship each column has on the impact."),                    
                            html.Div("Click on a column in the Impact graph to explore the visualization in the Impact Relationship graph below."),                    
                        ], style=dict(margin=10)), 
                        ], style=dict(margin=10)),
                    dbc.Row([
                        dbc.Col([
                            self.shap_summary.layout()
                        ], style=dict(margin=10)),
                    ], style=dict(margin=10)),
                    dbc.Row([            
                        dbc.Col([                    
                            self.shap_dependence.layout()
                        ], style=dict(margin=10)),
                    ], style=dict(margin=10)),
                ])

        class Interactions(ExplainerComponent):
            def __init__(self, explainer, name=None):
                super().__init__(explainer, title="Impact Interaction")
                self.interaction = InteractionSummaryComponent(explainer, name=self.name+"interaction",hide_depth=True)
                self.interactiond = InteractionDependenceComponent(explainer, name=self.name+"interactiond",hide_depth=True)
                self.interactioncon = InteractionSummaryDependenceConnector(self.interaction, self.interactiond)            

            def layout(self):
                return dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            html.H3("Deep Dive"),
                            html.Div("Explore the effect of"),                
                        ], style=dict(margin=10)), 
                        ], style=dict(margin=10)),
                    dbc.Row([
                        dbc.Col([
                            self.interaction.layout()
                        ], style=dict(margin=10)),
                    ]),
                    dbc.Row([            
                        dbc.Col([
                            html.H3("Feature impact plot"),
                            self.interactiond.layout()
                        ], style=dict(margin=10)),
                    ]),
                ])

        class Classif(ExplainerComponent):
            def __init__(self, explainer, name=None):
                super().__init__(explainer, title="Regression Metrics")
                self.predsum = RegressionModelSummaryComponent(explainer, name=self.name+"predsum",hide_cutoff= True,cutoff=0.5)
                self.interaction = PredictedVsActualComponent(explainer, name=self.name+"interaction",hide_cutoff= True,cutoff=0.5)
                self.interactiond = PrecisionComponent(explainer, name=self.name+"interactiond")         
                self.interaction1 = RegressionVsColComponent(explainer, name=self.name+"interaction1")         
                
            def layout(self):
                return dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            html.H3("Evaluate Regression Model Metrics"),
                            html.Div("Hover onto each metric to understand the regression model performance"),                                        
                        ], style=dict(margin=10)), 
                        ], style=dict(margin=10)),
                    dbc.Row([
                        dbc.Col([
                            self.predsum.layout()
                        ], style=dict(margin=10)),
                    ], style=dict(margin=10)),            
                    # dbc.Row([
                    #     dbc.Col([                              
                    #         self.interaction.layout()
                    #     ], style=dict(margin=10)),
                    # ], style=dict(margin=10)),            
                    dbc.Row([
                        dbc.Col([
                            self.interaction.layout()
                        ], style=dict(margin=10)),
                    ], style=dict(margin=10)),            
                    dbc.Row([
                        dbc.Col([
                            self.interaction1.layout()
                        ], style=dict(margin=10)),
                    ], style=dict(margin=10)),            
                ])


        ExplainerDashboard(explainer, [CustomDashboard, CustomPredictionsTab, Classif], boostrap=dbc.themes.FLATLY, title=projecttitle, plot_sample=1000).run()


else: 
    b = []
    for i in algo:
        model = i().fit(x_train, y_train.values.ravel())
  
        #testing training accuracy
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import explained_variance_score
        from sklearn.metrics import max_error
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import r2_score

        y_pred = model.predict(x_test)
        x = model.score(x_test,y_pred) 
        y = mean_squared_error(y_test, y_pred)
        z = explained_variance_score(y_test, y_pred)
        z1 = max_error(y_test, y_pred)
        z2 = mean_absolute_error(y_test, y_pred)
        # z3 = mean_squared_error(y_test, y_pred)
        z4 = r2_score(y_test, y_pred)
        # b.append(x - y + z - z1 - z2 + z4)
        v = z1+y+z2
        b.append((300 - (v))/300 / 0.01)
        print("Accuracy:", x ,"MSE:", y ,"EVS:", z ,"ME:", z1 ,"MAE:", z2 ,"R2:", z4)
       
    
        print(b)
        init_score = max(b)

        from sklearn import preprocessing
        import numpy as np
        b = preprocessing.normalize([b])
        smallest = np.min(b)
        #save best score as final best score in home page
        best_score=((1 - smallest) / 0.01)
        print(smallest)
        print(best_score)

        explainer = RegressionExplainer(model, x_test, y_test,                                 
                                cats=catCols,  
                                descriptions={},
                                units="$", shap_kwargs=dict(check_additivity=False))

        explainer.plot_contributions(0)

        class CustomDashboard(ExplainerComponent):
            def __init__(self, explainer, name=None, **kwargs):
                super().__init__(explainer, title="Impact")
                self.shap_summary = ShapSummaryComponent(explainer, name=self.name+"summary",
                                        hide_title=True, hide_depth=True)
                self.precision = PrecisionComponent(explainer, name=self.name+"precision",
                                        hide_cutoff=True, hide_binsize=True, 
                                        hide_binmethod=True, hide_multiclass=True,
                                        hide_selector=True, hide_title=True, hide_footer=True,
                                        cutoff=None, hide_depth=True)
                
                self.featuredesc = FeatureDescriptionsComponent(explainer, name=self.name+"featuredesc",hide_title=True, hide_subtitle=True)
                # self.connector = ShapSummaryDependenceConnector(self.shap_summary, self.shap_dependence)
                self.predictiontab = FeatureInputComponent(explainer, name=self.name+"predictiontab",title="What If..", subtitle="Adjust the column values to change the prediction", hide_index=False, hide_depth=False,fill_row_first=True)
                self.predictiongraph = ShapContributionsGraphComponent(explainer, name=self.name+"predictiongraph",hide_title=False, subtitle="How has each value contributed to the prediction?", hide_index=False, hide_depth=False, hide_sort=False, feature_input_component=self.predictiontab, hide_selector=False)
                self.predictioncontrib = ShapContributionsTableComponent(explainer, name=self.name+"predictioncontrib",hide_title=False, subtitle="How has each value contributed to the prediction?", hide_index=False, hide_depth=False, hide_sort=False, feature_input_component=self.predictiontab, hide_selector=False )
                self.predictionsum = ClassifierPredictionSummaryComponent(explainer, name=self.name+"predictionsum", **kwargs)
                self.predictionsum1 = ImportancesComponent(explainer, name=self.name+"predictionsum1", hide_type=True, hide_selector=True, hide_title=True, hide_subtitle=True)
                
                
            def layout(self):
                return dbc.Container([     
                    # dbc.Row([
                    #     dbc.Col([
                    #         html.H3("Feature Descriptions"),
                    #         self.featuredesc.layout()
                    #     ], style=dict(margin=20)),
                    #     ], style=dict(margin=20)),     
                    dbc.Row([
                        dbc.Col([
                            html.H3("Column Impact"),
                            html.Div("Analyze the impact each column has sorted from highest to lowest on the prediction."),
                            html.Div(f"{self.explainer.columns_ranked_by_shap()[0]} had the biggest impact"
                                    f", followed by {self.explainer.columns_ranked_by_shap()[1]}"
                                    f" and {self.explainer.columns_ranked_by_shap()[2]}."),                                                            
                        ], style=dict(margin=10)), 
                        ], style=dict(margin=10)),
                    dbc.Row([
                        dbc.Col([                    
                            self.predictionsum1.layout()
                        ], style=dict(margin=10)),
                        ], style=dict(margin=10)),        
                    dbc.Row([
                        dbc.Col([
                            html.H3("Individual Value Impact"),
                            html.Div("Explore the values from each column that have the greatest and least impact on the prediction."),
                            # self.predictiontab.layout()
                        ], style=dict(margin=10)),
                        ], style=dict(margin=10)),        
                    dbc.Row([
                        dbc.Col([                
                            self.predictiontab.layout()
                        ], style=dict(margin=10)),
                        ], style=dict(margin=10)),        
                    # dbc.Row([
                    #     dbc.Col([                    
                    #         self.predictiongraph.layout()
                    #     ], width=4, style=dict(margin=20)),
                    #     ]),
                    dbc.Row([
                        dbc.Col(self.predictioncontrib.layout(),style=dict(margin=10)),
                        dbc.Col(self.predictiongraph.layout(),style=dict(margin=10)),
                        # dbc.Col([                    
                        #     self.predictiongraph.layout()
                        # ]),
                        ],style=dict(margin=10)),
                    # dbc.Row([
                    #     dbc.Col([
                    #         html.H3("Prediction Summary"),
                    #         self.predsum.layout()
                    #     ], style=dict(margin=20)),
                    #     ]),
                ])

        class CustomPredictionsTab(ExplainerComponent):
            def __init__(self, explainer, name=None):
                super().__init__(explainer, title="Impact Relationship")
                self.shap_summary = ShapSummaryComponent(explainer, name=self.name+"summary",
                                    hide_title=True, hide_subtitle=True)
                self.shap_dependence = ShapDependenceComponent(explainer, name=self.name+"dependence", title="Impact Relationship", subtitle="Relationship between Feature value and Impact value")
                self.connector = ShapSummaryDependenceConnector(self.shap_summary, self.shap_dependence)                                

            def layout(self):
                return dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            html.H3("Link between Columns & Impact"),
                            html.Div("Analyze the relationship each column has on the impact."),                    
                            html.Div("Click on a column in the Impact graph to explore the visualization in the Impact Relationship graph below."),                    
                        ], style=dict(margin=10)), 
                        ], style=dict(margin=10)),
                    dbc.Row([
                        dbc.Col([
                            self.shap_summary.layout()
                        ], style=dict(margin=10)),
                    ], style=dict(margin=10)),
                    dbc.Row([            
                        dbc.Col([                    
                            self.shap_dependence.layout()
                        ], style=dict(margin=10)),
                    ], style=dict(margin=10)),
                ])

        class Interactions(ExplainerComponent):
            def __init__(self, explainer, name=None):
                super().__init__(explainer, title="Impact Interaction")
                self.interaction = InteractionSummaryComponent(explainer, name=self.name+"interaction",hide_depth=True)
                self.interactiond = InteractionDependenceComponent(explainer, name=self.name+"interactiond",hide_depth=True)
                self.interactioncon = InteractionSummaryDependenceConnector(self.interaction, self.interactiond)            

            def layout(self):
                return dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            html.H3("Deep Dive"),
                            html.Div("Explore the effect of"),                
                        ], style=dict(margin=10)), 
                        ], style=dict(margin=10)),
                    dbc.Row([
                        dbc.Col([
                            self.interaction.layout()
                        ], style=dict(margin=10)),
                    ]),
                    dbc.Row([            
                        dbc.Col([
                            html.H3("Feature impact plot"),
                            self.interactiond.layout()
                        ], style=dict(margin=10)),
                    ]),
                ])

        class Classif(ExplainerComponent):
            def __init__(self, explainer, name=None):
                super().__init__(explainer, title="Regression Metrics")
                self.predsum = RegressionModelSummaryComponent(explainer, name=self.name+"predsum",hide_cutoff= True,cutoff=0.5)
                self.interaction = PredictedVsActualComponent(explainer, name=self.name+"interaction",hide_cutoff= True,cutoff=0.5)
                self.interactiond = PrecisionComponent(explainer, name=self.name+"interactiond")         
                self.interaction1 = RegressionVsColComponent(explainer, name=self.name+"interaction1")         
                
            def layout(self):
                return dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            html.H3("Evaluate Regression Model Metrics"),
                            html.Div("Hover onto each metric to understand the regression model performance"),                                        
                        ], style=dict(margin=10)), 
                        ], style=dict(margin=10)),
                    dbc.Row([
                        dbc.Col([
                            self.predsum.layout()
                        ], style=dict(margin=10)),
                    ], style=dict(margin=10)),            
                    # dbc.Row([
                    #     dbc.Col([                              
                    #         self.interaction.layout()
                    #     ], style=dict(margin=10)),
                    # ], style=dict(margin=10)),            
                    dbc.Row([
                        dbc.Col([
                            self.interaction.layout()
                        ], style=dict(margin=10)),
                    ], style=dict(margin=10)),            
                    dbc.Row([
                        dbc.Col([
                            self.interaction1.layout()
                        ], style=dict(margin=10)),
                    ], style=dict(margin=10)),            
                ])

        ExplainerDashboard(explainer, [CustomDashboard, CustomPredictionsTab, Classif], boostrap=dbc.themes.FLATLY, title=projecttitle, plot_sample=1000).run()