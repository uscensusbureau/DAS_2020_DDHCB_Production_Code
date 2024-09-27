"""Domains for Pandas datatypes."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2023

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
from typeguard import check_type, typechecked

from tmlt.core.domains.base import Domain, OutOfDomainError
from tmlt.core.domains.numpy_domains import NumpyDomain


@dataclass(frozen=True)
class PandasSeriesDomain(Domain):
    """Domain of Pandas Series.

    Note:
        The index is always ignored when this domain type is used.
    """

    element_domain: NumpyDomain
    """Domain of elements in the Series."""

    def __post_init__(self):
        """Checks arguments to constructor."""
        check_type("element_domain", self.element_domain, NumpyDomain)

    @property
    def carrier_type(self) -> type:
        """Returns carrier type for members of the domain."""
        return pd.Series

    def validate(self, value: Any):
        """Raises error if value is not a DataFrame with matching schema."""
        # NOTE: Can not assert (elem in self.element_domain for elem in value) because
        # iterating over a Series implicitly calls item() on the NumPy values
        # retrieving the corresponding python object
        super().validate(value)
        for i in range(len(value)):  # pylint: disable=consider-using-enumerate
            try:
                self.element_domain.validate(value[i])
            except OutOfDomainError as exception:
                raise OutOfDomainError(
                    self, value, f"Found invalid value in Series: {exception}"
                ) from exception

    @classmethod
    def from_numpy_type(cls, dtype: np.dtype) -> "PandasSeriesDomain":
        """Returns a Pandas Series from a NumPy type."""
        return PandasSeriesDomain(NumpyDomain.from_np_type(dtype))


PandasColumnsDescriptor = Dict[str, PandasSeriesDomain]
"""Mapping from column name to column domain."""


class PandasDataFrameDomain(Domain):
    """Domain of Pandas DataFrames."""

    @typechecked
    def __init__(self, schema: PandasColumnsDescriptor):
        """Constructor.

        Args:
            schema: Mapping from column name to column domain.
        """
        self._schema = schema.copy()

    def __repr__(self) -> str:
        """Return string representation of the object."""
        return f"{self.__class__.__name__}(schema={self._schema})"

    @property
    def schema(self) -> PandasColumnsDescriptor:
        """Returns mapping from column name to associated domain."""
        return self._schema.copy()

    @property
    def carrier_type(self) -> type:
        """Returns carrier type for the domain."""
        return pd.DataFrame

    def validate(self, value: Any):
        """Raises error if value is not a Pandas DataFrame with matching schema."""
        super().validate(value)
        value_columns = list(value.columns)
        if len(value_columns) > len(set(value_columns)):
            duplicates = set(
                col for col in value_columns if value_columns.count(col) > 1
            )
            raise OutOfDomainError(
                self, value, f"Some columns are duplicated, {sorted(duplicates)}"
            )

        schema_columns = list(self.schema.keys())
        if value_columns != schema_columns:
            raise OutOfDomainError(
                self,
                value,
                (
                    "Columns are not as expected. DataFrame and Domain must contain"
                    " the same columns in the same order.\nDataFrame columns:"
                    f" {value_columns}\nDomain columns: {schema_columns}"
                ),
            )

        for column in self.schema:
            try:
                self.schema[column].validate(value[column])
            except OutOfDomainError as exception:
                raise OutOfDomainError(
                    self,
                    value,
                    f"Found invalid value in column '{column}': {exception}",
                ) from exception

    def __eq__(self, other: Any) -> bool:
        """Return True if the classes are equivalent."""
        if self.__class__ != other.__class__:
            return False
        return OrderedDict(self.schema) == OrderedDict(other.schema)

    @classmethod
    def from_numpy_types(cls, dtypes: Dict[str, np.dtype]) -> "PandasDataFrameDomain":
        """Returns a Pandas DataFrame domain from a dictionary of NumPy types."""
        col_to_desc = {}
        for col in dtypes:
            col_to_desc[col] = PandasSeriesDomain.from_numpy_type(dtypes[col])
        return PandasDataFrameDomain(col_to_desc)
