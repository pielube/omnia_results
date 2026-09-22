"""Explicit regional aggregation for the compact report figure collection.

Prices use fixed base-year emissions weights; reported intensities use each
year's production weights. Neither calculation fills missing source ratios.
"""

from collections.abc import Mapping
from numbers import Integral

import numpy as np
import pandas as pd

from .metrics import Results, SECTORS, Sector, ratio


class ReportData:
    def __init__(self, results: Results, config: dict):
        self.results = results
        self.macroregions = self._partition(config.get("macroregions"), "macroregions")
        self.producer_groups = self._partition(config.get("producer_groups"), "producer_groups")
        self.ranking_year = config.get("ranking_year", 2050)
        self.price_weight_year = config.get("price_weight_year", 2019)
        for name in ("ranking_year", "price_weight_year"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value not in results.years:
                raise ValueError(f"{name} must be an integer in results.years")
        self.top_n = config.get("top_n", 12)
        if isinstance(self.top_n, bool) or not isinstance(self.top_n, Integral) or self.top_n <= 0:
            raise ValueError("top_n must be a positive integer")
        self._cache: dict[tuple, pd.DataFrame] = {}

    def _partition(self, groups, name: str) -> dict[str, list[str]]:
        if not isinstance(groups, Mapping) or not groups:
            raise ValueError(f"{name} must be a nonempty mapping of group names to regions")
        copied = {}
        for label, members in groups.items():
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"{name} requires nonempty group names")
            if not isinstance(members, (list, tuple)) or not members:
                raise ValueError(f"{name}: group {label!r} must contain a nonempty list of regions")
            if not all(isinstance(region, str) for region in members):
                raise ValueError(f"{name}: region identifiers must be strings")
            copied[label] = list(members)
        flattened = [region for members in copied.values() for region in members]
        duplicates = sorted({region for region in flattened if flattened.count(region) > 1})
        missing = sorted(set(self.results.regions) - set(flattened))
        unexpected = sorted(set(flattened) - set(self.results.regions))
        if duplicates or missing or unexpected:
            raise ValueError(f"{name} must partition results.regions exactly once; "
                             f"duplicates={duplicates}, missing={missing}, unexpected={unexpected}")
        return copied

    @staticmethod
    def _sector(sector: Sector | str) -> Sector:
        if isinstance(sector, Sector):
            return sector
        for candidate in SECTORS:
            if sector == candidate.key:
                return candidate
        raise ValueError(f"Unknown sector: {sector!r}")

    def production_by_region(self, sector: Sector | str) -> pd.DataFrame:
        """Final material production in Mt/yr, excluding cement clinker."""
        sector = self._sector(sector)
        key = ("production", sector.key)
        if key not in self._cache:
            variable = f"Production|{sector.path}"
            if sector.key == "cement":
                data = self.results.values(variable, "Mt/yr")
            else:
                data = (self.results.values(variable + "|Primary", "Mt/yr")
                        + self.results.values(variable + "|Secondary", "Mt/yr"))
            self._cache[key] = data
        return self._cache[key].copy()

    def _group_sum(self, data: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
        output = pd.DataFrame({label: data.loc[members].sum(axis=0, min_count=len(members))
                               for label, members in groups.items()}).T
        output.index.name = "Group"
        output.columns.name = "Year"
        return output

    def production_groups(self, sector: Sector | str) -> pd.DataFrame:
        """Group production; one missing member makes the group's total unknown."""
        return self._group_sum(self.production_by_region(sector), self.producer_groups)

    def ranked_producers(self, sector: Sector | str) -> list[str]:
        """One fixed cohort, ranked by configured year with label-based tie breaking."""
        values = self.production_groups(sector)[self.ranking_year]
        eligible = values.loc[values.notna() & (values > 0)]
        return sorted(eligible.index, key=lambda label: (-eligible.loc[label], label))[:self.top_n]

    def producer_coverage(self, sector: Sector | str) -> pd.Series:
        groups = self.production_groups(sector)
        cohort = self.ranked_producers(sector)
        selected = groups.loc[cohort].sum(axis=0, min_count=len(cohort))
        global_production = self.results.aggregate(self.production_by_region(sector))
        return ratio(selected, global_production, 100).rename("GlobalShare_percent")

    def reported_intensity_groups(self, sector: Sector | str) -> pd.DataFrame:
        """Production-weighted reported energy intensities in GJ/t.

        A zero-production member contributes zero even when its intensity is
        absent. Any missing production or positive-production missing intensity
        keeps the aggregate undefined. Groups with zero production are undefined.
        """
        sector = self._sector(sector)
        production = self.production_by_region(sector)
        intensity = self.results.values(f"Energy Intensity|Industry|{sector.path}",
                                        "EJ/Mt", activity=False) * 1000
        numerator = (production * intensity).mask(production.eq(0), 0.0)
        numerator_groups = self._group_sum(numerator, self.producer_groups)
        denominator_groups = self._group_sum(production, self.producer_groups)
        return numerator_groups.div(denominator_groups.where(denominator_groups > 0))

    def carbon_price_weights(self) -> pd.DataFrame:
        """Normalized, fixed industry-GHG-emissions weights, with their source data."""
        key = ("carbon_price_weights",)
        if key not in self._cache:
            base = self.results.values("Emissions|GHG|Industry", "MtCO2e/yr",
                                       activity=False)[self.price_weight_year]
            if not np.isfinite(base.to_numpy(dtype=float)).all() or (base < 0).any():
                raise ValueError("Carbon-price base emissions must be finite and nonnegative "
                                 "for every source region; missing emissions are not zero-filled")
            rows = []
            for label, members in self.macroregions.items():
                group_sum = base.loc[members].sum()
                if not np.isfinite(group_sum) or group_sum <= 0:
                    raise ValueError(f"Carbon-price base emissions must sum to a positive finite "
                                     f"value in group {label!r}")
                for region in members:
                    rows.append({"Group": label, "Region": region,
                                 "WeightYear": self.price_weight_year,
                                 "Weight": base.loc[region] / group_sum,
                                 "BaseEmissions_MtCO2e": base.loc[region]})
            self._cache[key] = pd.DataFrame(rows)
        return self._cache[key].copy()

    def carbon_prices(self) -> pd.DataFrame:
        """Macroregional price indices using constant, normalized emissions weights."""
        weights = self.carbon_price_weights()
        prices = self.results.values("Price|Carbon", "USD_2010/t CO2e", activity=False)
        groups = {}
        for label in self.macroregions:
            group_weights = weights.loc[weights.Group == label].set_index("Region").Weight
            positive = group_weights.loc[group_weights > 0]
            weighted = prices.loc[positive.index].mul(positive, axis=0)
            groups[label] = weighted.sum(axis=0, min_count=len(positive))
        output = pd.DataFrame(groups).T
        output.index.name = "Group"
        output.columns.name = "Year"
        return output

    def membership(self) -> pd.DataFrame:
        return pd.DataFrame([{"GroupType": group_type, "Group": label, "Region": region}
                             for group_type, groups in (("macroregion", self.macroregions),
                                                        ("producer", self.producer_groups))
                             for label, members in groups.items() for region in members])

    def producer_ranking_table(self) -> pd.DataFrame:
        rows = []
        for sector in SECTORS:
            production = self.production_groups(sector)[self.ranking_year]
            global_total = self.results.aggregate(self.production_by_region(sector)).loc[self.ranking_year]
            for rank, label in enumerate(self.ranked_producers(sector), start=1):
                rows.append({"Sector": sector.label, "Rank": rank, "Group": label,
                             "RankingYear": self.ranking_year,
                             "Production_Mt_yr": production.loc[label],
                             "GlobalShare_percent": production.loc[label] / global_total * 100
                             if pd.notna(global_total) and global_total > 0 else np.nan})
        return pd.DataFrame(rows, columns=["Sector", "Rank", "Group", "RankingYear",
                                           "Production_Mt_yr", "GlobalShare_percent"])
