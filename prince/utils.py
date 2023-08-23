import functools

import altair as alt
import jax.numpy as np
from polars import DataFrame
from sklearn.utils import validation


def check_is_fitted(method):
    @functools.wraps(method)
    def _impl(self, *method_args, **method_kwargs):
        validation.check_is_fitted(self)
        return method(self, *method_args, **method_kwargs)

    return _impl


class EigenvaluesMixin:
    @property
    @check_is_fitted
    def percentage_of_variance_(self):
        """Returns the percentage of explained inertia per principal component."""
        return 100 * self.eigenvalues_ / self.total_inertia_

    @property
    @check_is_fitted
    def cumulative_percentage_of_variance_(self):
        """Returns the percentage of explained inertia per principal component."""
        return np.cumsum(self.percentage_of_variance_)

    @property
    @check_is_fitted
    def _eigenvalues_summary(self):
        """Return a summary of the eigenvalues and their importance."""
        return DataFrame(
            {
                "eigenvalue": self.eigenvalues_,
                r"% of variance": self.percentage_of_variance_,
                r"% of variance (cumulative)": self.cumulative_percentage_of_variance_,
            },
        )

    @property
    def eigenvalues_summary(self):
        """Return a summary of the eigenvalues and their importance."""
        summary = self._eigenvalues_summary
        summary["% of variance"] /= 100
        summary["% of variance (cumulative)"] /= 100
        summary["eigenvalue"] = summary["eigenvalue"].map("{:,.3f}".format)
        summary["% of variance"] = summary["% of variance"].map("{:.2%}".format)
        summary["% of variance (cumulative)"] = summary["% of variance (cumulative)"].map(
            "{:.2%}".format
        )
        summary.index.name = "component"
        return summary

    def scree_plot(self):
        """Scree plot.

        References
        ----------
        https://en.wikipedia.org/wiki/Scree_plot

        """
        eig = self._eigenvalues_summary.reset_index()
        eig["component"] = eig["component"].astype(str)
        return (
            alt.Chart(
                self._eigenvalues_summary.reset_index().assign(
                    component=lambda x: x["component"].astype(str)
                )
            )
            .mark_bar(size=10)
            .encode(x="component", y="eigenvalue", tooltip=eig.columns.tolist())
        )
