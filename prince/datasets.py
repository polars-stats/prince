import pathlib

import polars as pl

DATASETS_DIR = pathlib.Path(__file__).parent / "datasets"


def load_energy_mix(year=2019, normalize=True) -> pl.DataFrame:
    """Per capita energy mix by country in 2019.

    Each row corresponds to a country. There is one column for each energy source.
    A value corresponds to the average energy consumption of a source per capita.
    For instance, in France, every citizen consumed 15,186 kWh of nuclear energy.

    This data comes from https://ourworldindata.org/energy-mix

    Parameters
    ----------
    year
        The year the study was made.
    normalize
        Whether or not to normalize the kWh by country.

    """

    df = (
        pl.read_csv(DATASETS_DIR / "per-capita-energy-stacked.csv")
        .filter(pl.col('Year') == year)
        .filter(pl.col('Entity').is_not_in(['Africa', 'Europe', 'North America', 'World']))
        .drop(columns=["Code", "Year"])
        .rename(columns={"Entity": "Country"})
        .rename(columns=lambda x: x.replace(" per capita (kWh)", "").lower())
        .set_index(["continent", "country"])
    )
    if normalize:
        return df.div(df.sum(axis="columns"), axis="rows")
    return df


def load_decathlon():
    """The Decathlon dataset from FactoMineR."""
    decathlon = pl.read_csv(DATASETS_DIR / "decathlon.csv")
    decathlon.columns = ['athlete', *map(str.lower, decathlon.columns[1:])]
    decathlon = decathlon.with_columns(pl.col('athlete').apply(str.title))
    return decathlon


def load_french_elections():
    """Voting data for the 2022 French elections, by region.

    The [original dataset](https://www.data.gouv.fr/fr/datasets/resultats-du-premier-tour-de-lelection-presidentielle-2022-par-commune-et-par-departement/#resources)
    has been transformed into a contingency matrix. The latter tallies the number of votes for the
    12 candidates across all 18 regions. The number of blank and abstentions are also recorded.
    More information about these regions, including a map, can be found
    [on Wikipedia](https://www.wikiwand.com/fr/Région_française).

    """
    dataset = pl.read_csv(DATASETS_DIR / "02-resultats-par-region.csv")
    cont = (
        dataset.pivot(values='cand_nb_voix', index='reg_name', columns='cand_nom')
        .groupby('reg_name')
        .agg(
            pl.col('abstention_nb').min().alias('Abstention'),
            pl.col('blancs_nb').min().alias('Blank'),
        )
    )
    cont.columns = [c.title() for c in cont.columns]
    cont.index.name = "region"
    cont.columns.name = "candidate"
    return cont


def load_punctuation_marks():
    """Punctuation marks of six French writers."""
    return pl.read_csv(DATASETS_DIR / "punctuation_marks.csv")


def load_hearthstone_cards():
    """Hearthstone standard cards.

    Source: https://gist.github.com/MaxHalford/32ed2c80672d7391ec5b4e6f291f14c1

    """
    return pl.read_csv(DATASETS_DIR / "hearthstone_cards.csv")


def load_burgundy_wines():
    """Burgundy wines dataset.

    Source: https://personal.utdallas.edu/~herve/Abdi-MCA2007-pretty.pdf

    """
    wines = pl.DataFrame(
        data=[
            [1, 6, 7, 2, 5, 7, 6, 3, 6, 7],
            [5, 3, 2, 4, 4, 4, 2, 4, 4, 3],
            [6, 1, 1, 5, 2, 1, 1, 7, 1, 1],
            [7, 1, 2, 7, 2, 1, 2, 2, 2, 2],
            [2, 5, 4, 3, 5, 6, 5, 2, 6, 6],
            [3, 4, 4, 3, 5, 4, 5, 1, 7, 5],
        ],
        columns=pl.MultiIndex.from_tuples(
            [
                ("Expert 1", "Fruity"),
                ("Expert 1", "Woody"),
                ("Expert 1", "Coffee"),
                ("Expert 2", "Red fruit"),
                ("Expert 2", "Roasted"),
                ("Expert 2", "Vanillin"),
                ("Expert 2", "Woody"),
                ("Expert 3", "Fruity"),
                ("Expert 3", "Butter"),
                ("Expert 3", "Woody"),
            ],
            names=("expert", "aspect"),
        ),
        index=[f"Wine {i + 1}" for i in range(6)],
    )
    wines.insert(0, "Oak type", [1, 2, 2, 2, 1, 1])
    return wines


def load_beers():
    """Beers dataset.

    The data is taken from https://github.com/philipperemy/beer-dataset.

    """
    return pl.read_csv(DATASETS_DIR / "beers.csv.zip")
