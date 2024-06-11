"""Results visualization pane.

"""
from typing import Callable, List, Tuple, Union, Iterable

import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import display

# Plotting function that takes dataframe, model_col (str), label_col (str), pred_col (str)
PLOT_FXN =  Callable[[pd.DataFrame, str, str, str], None]
# Plotting function that takes dataframe, buckets (list of int/float tuples), bucket_col (str),
# bucket_display_str (str), model_col (str), label_col (str), pred_col (str)
PLOT_BY_BUCKET_FXN =  Callable[[pd.DataFrame, List[Tuple[Union[int, float]]], str, str, str, str, str], None]
# Plotting function that takes dataframe, annotation_col (str), model_col (str), label_col (str), pred_col (str)
PLOT_BY_ANNOTATION_FXN =  Callable[[pd.DataFrame, str, str, str, str], None]


class ResultsViz:
    # annotations: tuple = ('3UTR', '5UTR',
    #                       'CTCF-bound',
    #                       'K562_H2AFZ', 'K562_H3K27ac', 'K562_H3K27me3', 'K562_H3K36me3',
    #                       'K562_H3K4me1', 'K562_H3K4me2', 'K562_H3K4me3', 'K562_H3K9ac',
    #                       'K562_H3K9me3', 'K562_H4K20me1',
    #                       'TES', 'TSS',
    #                       'enhancer_Tissue_invariant', 'enhancer_Tissue_specific',
    #                       'exon', 'intron', 'lncRNA', 'polyA_signal', 'polyA_site',
    #                       'promoter_TATA', 'promoter_Tissue_invariant', 'promoter_Tissue_specific',
    #                       'promoter_non_TATA', 'protein_coding_gene', 'splice_acceptor', 'splice_donor')
    # models: tuple = ('enformer','dnabert_1', 'dnabert_2',
    #                  'hyena_1K', 'hyena_16K', 'hyena_32K', 'hyena_160K',
    #                  'dcnuc_50M', 'dcnuc_100M', 'dcnuc_250M', 'dcnuc_500M',
    #                  'dcnuc_50M_4K', 'dcnuc_50M_8K', 'dcnuc_500M_10K')

    def __init__(
        self,
        df: pd.DataFrame,
        base_models: Union[str, Iterable[str], None],
        annotations: Iterable[str],
        models: Iterable[str],
        plot_fxn: PLOT_FXN,
        plot_by_tss_dist_fxn: Union[PLOT_BY_BUCKET_FXN, None] = None,
        plot_by_annotation_fxn: Union[PLOT_BY_ANNOTATION_FXN, None] = None,
        plot_by_maf_fxn: Union[PLOT_BY_BUCKET_FXN, None] = None,
        distance_to = 'tss'
    ):
        
        self.df = df
        self.annotations = annotations
        self.models = tuple(models)
        # if base_models is not None:
            # base_models = tuple(base_models) if isinstance(base_models, Iterable) else (base_models,)
            # self.models = base_models + self.models
        self.filtered_df = self.df.copy()
        self.plot_fxn = plot_fxn
        self.plot_by_tss_dist_fxn = plot_by_tss_dist_fxn
        self.plot_by_annotation_fxn = plot_by_annotation_fxn
        self.plot_by_maf_fxn = plot_by_maf_fxn
        self.distance_to = distance_to
        
        model_header = widgets.HTML(description="Include Models:", value="",
                                    style={'description_width': 'initial'})
        self.model_check_boxes = {
            model: widgets.Checkbox(description=model, value=True, continuous_update=False,
                                    style={'description_width': 'initial'},
                                    layout=widgets.Layout(width='200px'))
            for model in self.models
        }
        for checkbox in self.model_check_boxes.values():
            checkbox.observe(self.filter_df_and_update_all_plots, names='value')

        select_all = widgets.Button(description='Select all')
        select_all.style.button_color = 'lightblue'
        select_all.on_click(self.select_all_fn)
        clear_all = widgets.Button(description='Clear all',)
        clear_all.on_click(self.clear_all_fn)
        
        model_widget = widgets.VBox([
            widgets.HBox([model_header, select_all, clear_all]),
            widgets.HBox([
                widgets.VBox([self.model_check_boxes[model]
                              for model in base_models]) if base_models is not None else widgets.VBox([]),
                widgets.VBox([self.model_check_boxes[model]
                              for model in self.models if 'DNABERT' in model]),
                widgets.VBox([self.model_check_boxes[model]
                              for model in self.models if 'Hyena' in model]),
                widgets.VBox([self.model_check_boxes[model]
                              for model in self.models if 'NTv2' in model and len(model.split(' ')) == 2]),
                widgets.VBox([self.model_check_boxes[model]
                              for model in self.models if 'NTv2' in model and len(model.split(' ')) == 3]),
             ], layout=widgets.Layout(width='100%'))
        ])
        
        # TODO: Currently hardcoding, perhaps make adjustable?

        if self.distance_to == 'tss':
            self.tss_dist_buckets = [(0, 10_000), (10_000, 100_000), (100_000, np.infty)]
        elif self.distance_to =='enhancer':
            self.tss_dist_buckets =[(0,1000),(1000,10000),(10000,np.infty)]
        self.maf_buckets = [(0.05, 0.075), (0.075, 0.1), (0.1, 0.15), (0.15, 0.25), (0.25, 0.35), (0.35, 0.5)]

        self.annotation_dropdown = widgets.Dropdown(
            options=self.annotations,
            description='Split by:'
        )
        self.annotation_dropdown.observe(self.update_plot_by_annotation, names='value')

        self.plot_output = widgets.Output()
        self.plot_by_tss_dist_output = widgets.Output()
        self.plot_by_annotation_output = widgets.Output()
        self.plot_by_maf_output = widgets.Output()
        
        plot_header = widgets.HTML(value="<h3>Aggregate Result</h3>")
        display_widgets = [model_widget, plot_header, self.plot_output]
        if self.plot_by_tss_dist_fxn is not None:
            display_widgets += [widgets.HTML(value=f"<h3>Result by Distance to {self.distance_to.capitalize()}</h3>"),
                                self.plot_by_tss_dist_output]

        if self.plot_by_annotation_fxn is not None:
            display_widgets += [widgets.HTML(value="<h3>Result by Annotation</h3>"), self.annotation_dropdown,
                                self.plot_by_annotation_output]
        if self.plot_by_maf_fxn is not None:
            display_widgets += [widgets.HTML(value="<h3>Result by MAF</h3>"), self.plot_by_maf_output]
        display(widgets.VBox(display_widgets))
        with self.plot_output:
            self.plot_fxn(df)
        with self.plot_by_annotation_output:
            if self.plot_by_annotation_fxn is not None:
                self.plot_by_annotation_fxn(df, self.annotation_dropdown.value)
        with self.plot_by_tss_dist_output:
            if self.plot_by_tss_dist_fxn is not None:
                self.plot_by_tss_dist_fxn(
                    df, buckets=self.tss_dist_buckets)
        with self.plot_by_maf_output:
            if self.plot_by_maf_fxn is not None:
                self.plot_by_maf_fxn(
                    df, buckets=self.maf_buckets)
        
    def filter_df_and_update_all_plots(self, value=None, model_col='model', **kwargs) -> None:
        self.filtered_df = self.df.copy()
        filter_models = [model for model, checkbox in self.model_check_boxes.items() if checkbox.value]
        self.filtered_df = self.filtered_df[self.filtered_df[model_col].isin(filter_models)]
        self.update_all_plots()

    def update_all_plots(self, _=None) -> None:
        self.update_plot()
        self.update_plot_by_tss_dist()
        self.update_plot_by_annotation()
        self.update_plot_by_maf()

    def update_plot(self) -> None:        
        self.plot_output.clear_output()
        if len(self.filtered_df) > 0:
            with self.plot_output:
                self.plot_fxn(self.filtered_df)
    
    def update_plot_by_tss_dist(self, _=None) -> None:
        self.plot_by_tss_dist_output.clear_output()
        if len(self.filtered_df) > 0:
            with self.plot_by_tss_dist_output:
                if self.plot_by_tss_dist_fxn is not None:
                    self.plot_by_tss_dist_fxn(self.filtered_df, buckets=self.tss_dist_buckets)

    def update_plot_by_annotation(self, _=None,) -> None:
        self.plot_by_annotation_output.clear_output()
        if len(self.filtered_df) > 0:
            with self.plot_by_annotation_output:
                if self.plot_by_annotation_fxn is not None:
                    self.plot_by_annotation_fxn(self.filtered_df, self.annotation_dropdown.value)

    def update_plot_by_maf(self, _=None,) -> None:
        self.plot_by_maf_output.clear_output()
        if len(self.filtered_df) > 0:
            with self.plot_by_maf_output:
                if self.plot_by_maf_fxn is not None:
                    self.plot_by_maf_fxn(self.filtered_df, buckets=self.maf_buckets)

    def select_all_fn(self, _=None) -> None:
        for checkbox in self.model_check_boxes.values():
            checkbox.unobserve(self.filter_df_and_update_all_plots, names='value')
            checkbox.value = True
            checkbox.observe(self.filter_df_and_update_all_plots, names='value')
        self.filter_df_and_update_all_plots()
        
        
    def clear_all_fn(self, _=None) -> None:
        for checkbox in self.model_check_boxes.values():
            checkbox.unobserve(self.filter_df_and_update_all_plots, names='value')
            checkbox.value = False
            checkbox.observe(self.filter_df_and_update_all_plots, names='value')
        self.filter_df_and_update_all_plots()
