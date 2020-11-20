"""covidtracker

A convenient plotter for the United States COVID-19 Cases and Deaths

Copyright 2020 Grant Lanham Jr

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.  This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details.  You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from pandas import DataFrame
import io
import numpy as np
import pandas as pd
import requests
import typing as tp

T = tp.TypeVar('T')

def get_dataframes() -> tp.Tuple[DataFrame, DataFrame, DataFrame]:
    req = requests.get(
        "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"
    )
    if req.ok:
        county_df = pd.read_csv(io.StringIO(req.content.decode('utf8')))

        county_df['county'] = county_df['county'].apply(str.upper)
        county_df['state'] = county_df['state'].apply(str.upper)
        county_df['date'] = pd.to_datetime(county_df['date'])

        # TODO: Only append the state abbreviation that have duplicated counties
        # county_count_df = county_df[['state', 'county', 'cases']].groupby(
        #     ['state', 'county'],
        #     as_index=False).first().groupby(['county']).count()['state'].reset_index()

        # dup_counties_df = county_count_df[
        #     county_count_df['state'] > 1].reset_index()

        # update_df = dup_counties_df[['county']].set_index('county').join(county_df.set_index('county')).reset_index()
        # update_df['county_new'] = update_df['county'] + '-' + update_df['state'].apply(lambda s: STATE_ABVS[s])

        # Simple solution for duplicated counties
        county_df['county'] = county_df['county'] + '-' + county_df[
            'state'].apply(lambda s: STATE_ABVS[s])

        state_df = county_df[['date', 'state', 'cases',
                              'deaths']].groupby(['date', 'state'
                                                  ]).agg('sum').reset_index()
        us_df = state_df[['date', 'cases', 'deaths'
                          ]].groupby(['date']).agg('sum').reset_index()
        return (county_df, state_df, us_df)
    else:
        return None


def get_threshold_date(days_back: tp.Union[None, int],
                       months_back: tp.Union[None, int]) -> datetime:
    days_back = 0 if days_back is None else days_back
    months_back = 0 if months_back is None else months_back
    total = days_back + months_back * 30
    if total <= 0:
        return datetime.min
    return datetime.today() - timedelta(days=total)


def df_rolling_average(df: DataFrame, columns: tp.List[str],
                       rolling_avg: int) -> DataFrame:
    df_ret = df.copy(deep=True)
    df_ret[columns] = df_ret[columns].rolling(rolling_avg).mean()
    return df_ret


def df_delta(df: DataFrame, columns: tp.List[str]) -> DataFrame:
    df_ret = df.copy(deep=True)
    df_ret[columns] = df_ret[columns].diff().shift(-1)
    return df_ret


def df_split_column(df: DataFrame, column: str, values: tp.List[str]):
    cnt = len(values)
    res = np.zeros(cnt, dtype=object)
    for i in range(cnt):
        filt = df[column] == values[i]
        res[i] = df[filt].reset_index()
    return res


def get_roll_avg_title(title: str, roll_avg: tp.Union[None, int],
                       delta: bool) -> str:
    if delta:
        title += ', Delta'
    if roll_avg is None:
        return title
    return f'{title}, {roll_avg} Rolling Average'


class CovidTracker:
    def __init__(self, county_df: DataFrame, state_df: DataFrame,
                 us_df: DataFrame):
        self.county_df = county_df
        self.state_df = state_df
        self.us_df = us_df
        self._counties = None
        self._states = None

    @classmethod
    def get_latest(cls):
        dfs = get_dataframes()
        if (dfs is None):
            raise ValueError("Error in getting latest dataframes")
        return cls(*dfs)

    @classmethod
    def copy(cls, covid_tracker):
        instance = cls(covid_tracker.county_df, covid_tracker.state_df,
                       covid_tracker.us_df)
        instance._counties = covid_tracker._counties
        instance._states = covid_tracker._states

    def _df_rolling_average(self, df: DataFrame, roll_avg: int) -> DataFrame:
        return df_rolling_average(df, ['cases', 'deaths'], roll_avg)

    def _df_delta(self, df: DataFrame):
        return df_delta(df, ['cases', 'deaths'])

    def _get_df(self, df: DataFrame, rolling_avg: tp.Union[None, int],
                delta: bool, days_back: int, months_back: int) -> DataFrame:
        tmp = df
        thresh_date = get_threshold_date(days_back, months_back)
        if thresh_date is not datetime.min:
            date_filter = df['date'] > get_threshold_date(
                days_back, months_back)
            tmp = df[date_filter]

        if rolling_avg is not None:
            tmp = self._df_rolling_average(tmp, rolling_avg)

        if delta:
            tmp = self._df_delta(tmp)

        return tmp

    def _get_axes(self, rolling_avg: tp.Union[None, int], delta: bool,
                  title: str) -> tp.Tuple[plt.Axes, plt.Axes]:

        title = get_roll_avg_title(title, rolling_avg, delta)

        if delta:
            y_case_label = 'Delta Cases'
            y_death_label = 'Delta Deaths'
        else:
            y_case_label = 'Cases'
            y_death_label = 'Deaths'

        fig, (ax_case, ax_death) = plt.subplots(1, 2)
        fig.set_size_inches(14.5, 6.5)
        ax_case.set_ylabel(y_case_label)
        ax_death.set_ylabel(y_death_label)
        fig.suptitle(title)
        return (ax_case, ax_death)

    def _plotter(self,
                 df: DataFrame,
                 ax_case: plt.Axes,
                 ax_death: plt.Axes,
                 cases_label: str,
                 death_label: str,
                 case_color='b',
                 death_color='r') -> None:
        x_dates = df['date'].values
        y_cases = df['cases'].values
        y_deaths = df['deaths'].values
        ax_case.plot(x_dates, y_cases, c=case_color, label=cases_label)
        ax_case.set_xlabel('Date')
        ax_case.legend()
        ax_case.xaxis_date()

        ax_death.plot(x_dates, y_deaths, c=death_color, label=death_label)
        ax_death.set_xlabel('Date')
        ax_death.legend()
        ax_death.xaxis_date()

        ax_death.get_figure().autofmt_xdate()

    def _plotter_us(self, roll_avg_days: tp.Union[None, int], delta: bool,
                    days_back: int, months_back: int) -> None:
        us_df_ra = self._get_df(self.us_df, roll_avg_days, delta, days_back,
                                months_back)

        ax_case, ax_death = self._get_axes(roll_avg_days, delta,
                                           'US COVID-19 Cases and Deaths')

        self._plotter(us_df_ra, ax_case, ax_death, None, None)

    def _plotter_columns(self, df: DataFrame, column: str,
                         values: tp.Union[tp.List[str], str],
                         roll_avg_days: tp.Union[None, int], delta: bool,
                         days_back: int, months_back: int, title: str):

        if isinstance(values, str):
            values = [values]

        ax_case, ax_death = self._get_axes(roll_avg_days, delta, title)

        case_colors = [
            'blue', 'navy', 'darkviolet', 'purple', 'magenta', 'cornflowerblue'
        ]
        death_colors = [
            'red', 'maroon', 'orangered', 'peru', 'goldenrod', 'rosybrown'
        ]
        max_len = len(case_colors)
        color_iter = 0
        for df_split in df_split_column(df, column, values):
            if color_iter > max_len - 1:
                color_iter = 0

            name = df_split[column][0]
            df_tmp = self._get_df(df_split, roll_avg_days, delta, days_back,
                                  months_back)
            self._plotter(df_tmp, ax_case, ax_death, name, name,
                          case_colors[color_iter], death_colors[color_iter])
            color_iter += 1

    def _plotter_counties(self, counties: tp.Union[tp.List[str], str],
                          roll_avg_days: tp.Union[None, int], delta: bool,
                          days_back: tp.Union[None, int],
                          months_back: tp.Union[None, int]):

        if isinstance(counties, str):
            counties = [counties]

        counties = map(lambda c: c.upper().strip(), counties)
        counties = list(filter(lambda c: c in self.counties, counties))
        self._plotter_columns(self.county_df, 'county', counties,
                              roll_avg_days, delta, days_back, months_back,
                              'COVID-19 US County Deaths and Cases')

    def _plotter_states(self, states: tp.Union[tp.List[str], str],
                        roll_avg_days: tp.Union[None, int], delta: bool,
                        days_back: tp.Union[None,
                                            int], months_back: tp.Union[None,
                                                                        int]):

        if isinstance(states, str):
            states = [states]

        states = map(lambda c: c.upper().strip(), states)
        states = list(filter(lambda c: c in self.states, states))
        self._plotter_columns(self.state_df, 'state', states, roll_avg_days,
                              delta, days_back, months_back,
                              'COVID-19 US State Deaths and Cases')

    def plot_states(self,
                    states: tp.Union[tp.List[str], str],
                    roll_avg_days: tp.Union[None, int] = None,
                    delta: bool = False,
                    days_back: tp.Union[None, int] = None,
                    months_back: tp.Union[None, int] = None):
        self._plotter_states(states, roll_avg_days, delta, days_back,
                             months_back)
        plt.show()

    def plot_counties(self,
                      counties: tp.Union[tp.List[str], str],
                      roll_avg_days: tp.Union[None, int] = None,
                      delta: bool = False,
                      days_back: tp.Union[None, int] = None,
                      months_back: tp.Union[None, int] = None):
        self._plotter_counties(counties, roll_avg_days, delta, days_back,
                               months_back)
        plt.show()

    def plot_us(self,
                roll_avg_days: tp.Union[None, int] = None,
                delta: bool = False,
                days_back: tp.Union[None, int] = None,
                months_back: tp.Union[None, int] = None) -> None:
        self._plotter_us(roll_avg_days, delta, days_back, months_back)
        plt.show()

    def plot(self,
             counties: tp.Union[tp.List[str], str, None] = None,
             states: tp.Union[tp.List[str], str, None] = None,
             roll_avg_days: tp.Union[None, int] = None,
             delta: bool = False,
             days_back: tp.Union[None, int] = None,
             months_back: tp.Union[None, int] = None,
             plot_us: bool = False) -> None:
        # NOTE: Colors
        # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
        counties_none = counties is None
        states_none = states is None

        if counties_none is False:
            self._plotter_counties(counties, roll_avg_days, delta, days_back,
                                   months_back)
        if states_none is False:
            self._plotter_states(states, roll_avg_days, delta, days_back,
                                 months_back)

        if plot_us or (counties_none and states_none):
            self._plotter_us(roll_avg_days, delta, days_back, months_back)

        plt.show()

    @property
    def states(self):
        if self._states is None:
            self._states = self.state_df['state'].unique()
        return self._states

    @property
    def counties(self):
        if self._counties is None:
            self._counties = self.county_df['county'].unique()
        return self._counties

    def _search(self, query: str, xs: tp.Iterable[str]) -> tp.Iterable:
        query = query.upper()
        return list(filter(lambda x: query in x.upper(), xs))

    def search_counties(self, query: str):
        return self._search(query, self.counties)

    def search_states(self, query: str):
        return self._search(query, self.states)
