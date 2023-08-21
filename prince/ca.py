"""Correspondence Analysis (CA)"""

import functools

import altair as alt
import numpy as np
import polars as pl
from polars import DataFrame, Series
from scipy import sparse
from sklearn.utils import check_array

from prince import svd, utils


def select_active_columns(method):
    @functools.wraps(method)
    def _impl(self, X=None, *method_args, **method_kwargs):
        if hasattr(self, "active_cols_") and isinstance(X, DataFrame):
            return method(self, X[self.active_cols_], *method_args, **method_kwargs)
        return method(self, X, *method_args, **method_kwargs)

    return _impl


def select_active_rows(method):
    @functools.wraps(method)
    def _impl(self, X=None, *method_args, **method_kwargs):
        if hasattr(self, "active_rows_") and isinstance(X, DataFrame):
            return method(self, X.loc[self.active_rows_], *method_args, **method_kwargs)
        return method(self, X, *method_args, **method_kwargs)

    return _impl


class CA(utils.EigenvaluesMixin):
    def __init__(
        self,
        n_components=2,
        n_iter=10,
        check_input=True,
        random_state=None,
        engine="sklearn",
    ):
        self.n_components = n_components
        self.n_iter = n_iter
        self.check_input = check_input
        self.random_state = random_state
        self.engine = engine

    def fit(self, X: DataFrame, y=None):
        # Check input
        if self.check_input:
            check_array(X)

        # Check all values are positive
        if (X < 0).select(pl.any_horizontal(pl.all())).to_series().any():
            raise ValueError("All values in X should be positive")

        X = X.to_numpy()

        # Compute the correspondence matrix which contains the relative frequencies
        X = X.astype(float) / np.sum(X)

        # Compute row and column masses
        self.row_masses_ = Series(X.sum(axis=1))
        self.col_masses_ = Series(X.sum(axis=0))

        self.active_rows_ = self.row_masses_.index.unique()
        self.active_cols_ = self.col_masses_.index.unique()

        # Compute standardised residuals
        r = self.row_masses_.to_numpy()
        c = self.col_masses_.to_numpy()
        S = sparse.diags(r**-0.5) @ (X - np.outer(r, c)) @ sparse.diags(c**-0.5)

        # Compute SVD on the standardised residuals
        self.svd_ = svd.compute_svd(
            X=S,
            n_components=min(self.n_components, min(X.shape) - 1),
            n_iter=self.n_iter,
            random_state=self.random_state,
            engine=self.engine,
        )

        # Compute total inertia
        self.total_inertia_ = np.einsum("ij,ji->", S, S.T)

        self.row_contributions_ = DataFrame(
            sparse.diags(self.row_masses_.values)
            @ (
                # Same as row_coordinates(X)
                (
                    sparse.diags(self.row_masses_.values**-0.5)
                    @ self.svd_.U
                    @ sparse.diags(self.svd_.s)
                )
                ** 2
            )
            / self.eigenvalues_,
        )

        self.column_contributions_ = DataFrame(
            sparse.diags(self.col_masses_.values)
            @ (
                # Same as col_coordinates(X)
                (
                    sparse.diags(self.col_masses_.values**-0.5)
                    @ self.svd_.V.T
                    @ sparse.diags(self.svd_.s)
                )
                ** 2
            )
            / self.eigenvalues_,
        )

        return self

    @property
    @utils.check_is_fitted
    def eigenvalues_(self):
        """Returns the eigenvalues associated with each principal component."""
        return np.square(self.svd_.s)

    @select_active_columns
    def row_coordinates(self, X: DataFrame) -> DataFrame:
        """The row principal coordinates."""

        try:
            X = X.sparse.to_coo().astype(float)
        except AttributeError:
            X = X.to_numpy()

        # Normalise the rows so that they sum up to 1
        X = X / X.sum(axis=1)

        return DataFrame(
            X @ sparse.diags(self.col_masses_.to_numpy() ** -0.5) @ self.svd_.V.T,
        )

    @select_active_columns
    def row_cosine_similarities(self, X: DataFrame):
        """Return the cos2 for each row against the dimensions.

        The cos2 value gives an indicator of the accuracy of the row projection on the dimension.

        Values above 0.5 usually means that the row is relatively accurately well projected onto that dimension. Its often
        used to identify which factor/dimension is important for a given element as the cos2 can be interpreted as the proportion
        of the variance of the element attributed to a particular factor.

        """
        F = self.row_coordinates(X)
        return self._row_cosine_similarities(X, F)

    @select_active_columns
    def _row_cosine_similarities(self, X: DataFrame, F):
        # Active
        X_act = X.loc[self.active_rows_]
        X_act = X_act / X_act.sum().sum()
        marge_col = X_act.sum(axis=0)
        Tc = X_act.div(X_act.sum(axis=1), axis=0).div(marge_col, axis=1) - 1
        dist2_row = (Tc**2).mul(marge_col, axis=1).sum(axis=1)

        # Supplementary
        X_sup = X.loc[X.index.difference(self.active_rows_, sort=False)]
        X_sup = X_sup.div(X_sup.sum(axis=1), axis=0)
        dist2_row_sup = ((X_sup - marge_col) ** 2).div(marge_col, axis=1).sum(axis=1)

        dist2_row = pl.concat((dist2_row, dist2_row_sup))

        # Can't use pandas.div method because it doesn't support duplicate indices
        return F**2 / dist2_row.to_numpy()[:, None]

    @select_active_rows
    def column_coordinates(self, X: DataFrame) -> DataFrame:
        """The column principal coordinates."""

        is_sparse = X.dtypes.apply(pl.api.types.is_sparse).all()
        if is_sparse:
            X = X.sparse.to_coo()
        else:
            X = X.to_numpy()

        # Transpose and make sure the rows sum up to 1
        X = X.T / X.T.sum(axis=1)

        return DataFrame(
            X @ sparse.diags(self.row_masses_.to_numpy() ** -0.5) @ self.svd_.U,
        )

    @select_active_rows
    def column_cosine_similarities(self, X: DataFrame):
        """Return the cos2 for each column against the dimensions.

        The cos2 value gives an indicator of the accuracy of the column projection on the dimension.

        Values above 0.5 usually means that the column is relatively accurately well projected onto that dimension. Its often
        used to identify which factor/dimension is important for a given element as the cos2 can be interpreted as the proportion
        of the variance of the element attributed to a particular factor.
        """
        G = self.column_coordinates(X)
        return self._column_cosine_similarities(X, G)

    @select_active_rows
    def _column_cosine_similarities(self, X, G):
        # Active
        X_act = X[self.active_cols_]
        X_act = X_act / X_act.sum().sum()
        marge_row = X_act.sum(axis=1)
        Tc = X_act.div(marge_row, axis=0).div(X_act.sum(axis=0), axis=1) - 1
        dist2_col = (Tc**2).mul(marge_row, axis=0).sum(axis=0)

        # Supplementary
        X_sup = X.select(pl.exclude(self.active_cols_))
        X_sup = X_sup.div(X_sup.sum(axis=0), axis=1)
        dist2_col_sup = ((X_sup.sub(marge_row, axis=0)) ** 2).div(marge_row, axis=0).sum(axis=0)

        dist2_col = pl.concat((dist2_col, dist2_col_sup))
        return (G**2).div(dist2_col, axis=0)

    @utils.check_is_fitted
    def plot(self, X: DataFrame, x_component=0, y_component=1, **params):
        row_coords = self.row_coordinates(X)
        row_coords.columns = [f"component {i}" for i in row_coords.columns]
        row_coords = row_coords.assign(
            variable=row_coords.index.name or "row", value=row_coords.index.astype(str)
        )

        col_coords = self.column_coordinates(X)
        col_coords.columns = [f"component {i}" for i in col_coords.columns]
        col_coords = col_coords.assign(
            variable=col_coords.index.name or "column",
            value=col_coords.index.astype(str),
        )

        coords = pl.concat([row_coords, col_coords])
        eig = self._eigenvalues_summary.to_dict(orient="index")

        return (
            alt.Chart(coords)
            .mark_circle(size=50)
            .encode(
                alt.X(
                    f"component {x_component}",
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(
                        title=f"component {x_component} — {eig[x_component]['% of variance'] / 100:.2%}"
                    ),
                ),
                alt.Y(
                    f"component {y_component}",
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(
                        title=f"component {y_component} — {eig[y_component]['% of variance'] / 100:.2%}"
                    ),
                ),
                color="variable",
                tooltip=[
                    "variable",
                    "value",
                    f"component {x_component}",
                    f"component {y_component}",
                ],
                **params,
            )
            .interactive()
        )
