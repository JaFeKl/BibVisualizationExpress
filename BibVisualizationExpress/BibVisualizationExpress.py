import os
import numpy as np
import csv
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pybliometrics.scopus import ScopusSearch, AbstractRetrieval
from timeit import default_timer as timer
from wordcloud import WordCloud


class BibVisualizationExpress():
    """
    A class for visualizing bibliographic data.

    This class provides methods to visualize bibliographic data, currently this class provides three plots:

    plot_records_over_time()
    plot_world_map_country_count()
    plot_keyword_worldcloud()
    """
    def __init__(self, cmap=plt.get_cmap('Blues')):
        """
        Initialize the BibExpressVisualization instance.

        Parameters:
        - cmap (matplotlib.colors.Colormap): The colormap to be used for visualizations.
          Default is 'Blues'.
        """
        self.df: pd.DataFrame = None
        self.module_path = os.path.dirname(__file__)
        self.pkg_path = os.path.dirname(self.module_path)
        self.csv_dir = os.path.join(self.pkg_path, "csv")
        self.cmap = cmap
        self.country_dict = self._load_country_dict()

    def _load_country_dict(self):
        """
        Load a dictionary mapping country names to ISO 3166-1 alpha-3 country codes.

        This method reads data from a CSV file containing country names and ISO 3166-1 alpha-3 codes,
        and constructs a dictionary mapping country names to their corresponding alpha-3 codes.

        Returns:
        - dict: A dictionary mapping country names to ISO 3166-1 alpha-3 country codes.
        """
        dic = {}
        print(os.path.join(self.module_path, "wikipedia-iso-country-codes.csv"))
        with open(os.path.join(self.module_path, "wikipedia-iso-country-codes.csv")) as f:
            file = csv.DictReader(f, delimiter=',')
            for line in file:
                dic[line['English short name lower case']] = line['Alpha-3 code']
        return dic

    def save(self, filename: str):
        """
        Save the DataFrame to a CSV file.

        This method resets the index of the DataFrame to ensure proper indexing, then saves it
        to a CSV file with the specified filename in the 'results' directory.

        Parameters:
        - filename (str): The name of the CSV file to save.

        """
        self.df = self.df.reset_index(drop=True)
        self.df.to_csv(os.path.join(self.csv_dir, filename))

    def load_csv(self, filename: str):
        """
        Load a DataFrame from a CSV file.

        This method loads a DataFrame from a CSV file with the specified filename located in the 'results' directory.
        It also resets the index of the DataFrame to ensure proper indexing.

        Parameters:
        - filename (str): The name of the CSV file to load.

        """
        self.df = pd.read_csv(os.path.join(self.csv_dir, filename))
        self.df = self.df.reset_index(drop=True)

    def search_scopus(self, query: str, postprocessing: bool = False) -> pd.DataFrame:
        """
        Search the Scopus database using a specified query.

        This method performs a search query on the Scopus database using the provided query string.
        It retrieves the search results and returns them as a pandas DataFrame.

        Parameters:
        - query (str): The search query to be performed.
        - postprocessing (bool) : Execute default postprocessing routines

        Returns:
        - pd.DataFrame: A DataFrame containing the search results retrieved from Scopus.

        """
        start = timer()
        s = ScopusSearch(query)
        self.df = pd.DataFrame(pd.DataFrame(s.results))
        end = timer()
        print(f"Retrieved {len(self.df)} initial results after {end-start} seconds")
        if postprocessing is True:
            self.remove_empty_doi()
            self.remove_record_types(['Conference Review'])
        return self.df

    def remove_record_types(self, types: list):
        """
        Remove records of specified types from the DataFrame.

        This method removes records of specified types from the DataFrame based on the 'subtypeDescription' column.
        Common types are "Article", "Conference Paper", "Conference Review", "Book Chapter", ...

        Parameters:
        - types (list): A list of record types to be removed.

        """
        previous_length = len(self.df)
        self.df = self.df[~self.df['subtypeDescription'].isin(types)]
        new_length = len(self.df)
        print(f"Removed record types {types}: "
              f"Removed {previous_length-new_length} records, "
              f"current number of records: {new_length}")

    def remove_empty_doi(self):
        """
        Remove records with empty DOI from the DataFrame.

        This method removes records from the DataFrame that have empty values in the 'doi' column.

        """
        previous_length = len(self.df)
        self.df = self.df.dropna(subset=["doi"])
        new_length = len(self.df)
        print(f"Removed records with empty DOI: "
              f"Removed {previous_length-new_length} records, "
              f"current number of records: {new_length}")

    def restrict_languages(self, languages: list):
        """
        Restrict DataFrame to include only records with specified languages.

        This method retrieves the language of each record based on its 'eid' (electronic identifier)
        using the '_retrieve_language' method and filters the DataFrame to include only records
        with languages specified in the 'languages' list.
        This method may take a long time, since the AbstractRetrieval API is used to get the languages
        Example language codes include 'eng' for English and 'ger' for German.

        Parameters:
        - languages (list): A list of language codes (ISO 639-2/T), a three letter code to restrict the DataFrame to.
        """
        self.df['language'] = self.df['eid'].apply(self._retrieve_language_of_record)
        previous_length = len(self.df)
        self.df = self.df[self.df['language'].isin(languages)]
        new_length = len(self.df)
        print(f"Removed records with with other languages then {languages}: "
              f"Removed {previous_length-new_length} records, "
              f"current number of records: {new_length}")

    def _retrieve_language_of_record(self, eid: str) -> str:
        """
        Retrieve the language of a record identified by its electronic identifier (eid).

        This method fetches the language of a record from the Scopus database using its electronic identifier (eid).
        It queries the Scopus database for the full abstract view ('FULL') to obtain detailed information,
        including the language.

        Parameters:
        - eid (str): The electronic identifier (eid) of the record.

        Returns:
        - str: The language of the record identified by the electronic identifier (eid).

        """
        abs = AbstractRetrieval(eid, view='FULL')
        return abs.language

    def plot_records_over_time(self, width: float = 6.4, height: float = 4.8,
                               plot_title: str = None, start_year: int = None, end_year: int = None):
        """
        Plot the number of records over time.

        This method plots the number of records over time, categorized by publication type.
        It shows the count of journal articles, conference proceedings, and other types of records for each year.
        Additionally, it displays the cumulative number of records over the years on a secondary y-axis.

        Parameters:
        - width (float, optional): Width of the plot figure. Defaults to 6.4.
        - height (float, optional): Height of the plot figure. Defaults to 4.8.
        - plot_title (str, optional): Title of the plot. Defaults to None.
        - start_year (int, optional): The start year for filtering the data. Defaults to None (no filtering).
        - end_year (int, optional): The end year for filtering the data. Defaults to None (no filtering).

        """
        fig = plt.figure(figsize=(width, height), constrained_layout=False)
        axes = fig.add_subplot()

        # prepare data
        pub_df = self._extract_records_count_per_year(start_year, end_year)
        year_list = pub_df.index.tolist()

        width = 0.6
        # Plot bars
        axes.bar(year_list,  pub_df['Journals'], color=self.cmap(1.0), width=width,
                 linewidth=0.7, label='Journals', zorder=1)
        axes.bar(year_list, pub_df['Conference Proceedings'], bottom=pub_df['Journals'], color=self.cmap(0.7),
                 width=width, linewidth=0.7, label='Conference Proceedings', zorder=1)
        axes.bar(year_list, pub_df['Others'], bottom=pub_df['Journals']+pub_df['Conference Proceedings'],
                 color=self.cmap(0.4), width=width, linewidth=0.7, label='Others', zorder=1)

        # Set x-axis
        axes.set_xticks(year_list)
        axes.set_xticklabels(year_list, rotation=50)

        # Adjust x-axis tick frequency for large number of years
        if len(year_list) > 20:  # Change threshold as needed
            label_list = axes.get_xticklabels()
            label_list_reversed = label_list[::-1]
            lables_to_hide = label_list_reversed[1::2]   # select every second label
            for label in lables_to_hide:
                label.set_visible(False)

        # Set y-axis
        axes.set_ylim((0, pub_df['Total'].max() + (pub_df['Total'].max() / 2)))
        axes.yaxis.set_major_locator(MaxNLocator(integer=True))
        axes.set_ylabel('Publications per year')

        # Plot cumulateive number of publications on secondary y-axis
        axes2 = axes.twinx()
        axes2.plot(year_list, pub_df['Total'].cumsum(), color='black', linestyle='dashed', label='Cumulative')
        axes2.yaxis.set_major_locator(MaxNLocator(integer=True))
        axes2.set_ylabel('Cumulative number of publications')

        h1, l1 = axes.get_legend_handles_labels()
        h2, l2 = axes2.get_legend_handles_labels()

        axes.legend(reversed(h1 + h2), reversed(l1 + l2), loc='upper left')
        if plot_title:
            axes.set_title(plot_title)

    def _extract_records_count_per_year(self, start_year: int = None, end_year: int = None):
        """
        Extract the publication count per year from the DataFrame.

        This method calculates the publication count per year from the DataFrame, categorized by publication type.
        It aggregates the count of journal articles, conference proceedings, and other types of records for each year.

        Parameters:
        - start_year (int, optional): The start year for filtering the publication count. Defaults to None.
        - end_year (int, optional): The end year for filtering the publication count. Defaults to None.

        Returns:
        - pd.DataFrame: A DataFrame containing the publication count per year, categorized by publication type.
        The DataFrame has columns for Journals, Conference Proceedings, Others, and Total.

        """
        if 'coverDate' in self.df.columns:
            self.df['year'] = pd.to_datetime(self.df['coverDate']).dt.year
        else:
            print("Unable to retrieve the year in the given dataframe...")
            return None

        filtered_df = None
        if start_year is not None:
            filtered_df = self.df[(self.df['year'] >= start_year)]
        if end_year is not None:
            filtered_df = filtered_df[(filtered_df['year'] <= end_year)]
        if filtered_df is None:
            filtered_df = self.df

        # Create a sorted list of unique years
        sorted_years_list = sorted(self.df['year'].unique())

        # Create a new list with all years from min to max
        expanded_year_list = list(range(min(sorted_years_list), max(sorted_years_list) + 1))

        pub_df = pd.DataFrame(index=expanded_year_list, columns=['Journals', 'Conference Proceedings', 'Others', 'Total'])

        for year in expanded_year_list:
            journal_counts = self.df[(self.df['year'] == year) &
                                     (self.df['aggregationType'] == 'Journal')]['year'].value_counts().get(year, 0)
            conference_counts = self.df[(self.df['year'] == year) &
                                        (self.df['aggregationType'] == 'Conference Proceeding')]['year'].value_counts().get(year, 0)
            other_counts = self.df[(self.df['year'] == year) &
                                   (~self.df['aggregationType'].isin(['Journal', 'Conference Proceeding']))]['year'].value_counts().get(year, 0)
            pub_df.at[year, 'Journals'] = journal_counts
            pub_df.at[year, 'Conference Proceedings'] = conference_counts
            pub_df.at[year, 'Others'] = other_counts
        pub_df['Total'] = pub_df['Journals'] + pub_df['Conference Proceedings'] + pub_df['Others']
        return pub_df

    def plot_keyword_worldcloud(self, width: float = 5.5, height: float = 5.5,
                                max_words: int = 20, min_font_size: int = 4, max_font_size: int = None):
        fig = plt.figure(figsize=(width, height), constrained_layout=False)
        wcloud_width = int(width/5 * 900)
        wcloud_height = int(height/5 * 900)
        ax = fig.add_subplot()
        all_keywords = []
        for row in self.df['authkeywords']:
            if row is not None:
                keywords = [keyword.strip().replace(' ', '_').lower() for keyword in row.split('|')]
                all_keywords.extend(keywords)

        keywords_series = pd.Series(all_keywords)
        keyword_counts = keywords_series.value_counts()
        max_words = min(max_words, len(keyword_counts))        # set the maximum words to be used in the cloud
        keyword_counts = keyword_counts[:max_words]            # filter series according to max words
        keyword_dict = keyword_counts.to_dict()

        max_freq = keyword_counts.max()
        min_freq = keyword_counts.min()
        # Scale the frequencies to range from 0.5 to 1.0
        scaled_freqs = {word: (freq - min_freq) / (max_freq - min_freq) * 0.6 + 0.5 for word, freq in keyword_dict.items()}

        def my_color_function(word, font_size, position, orientation, random_state=None, **kwargs):
            rgba_value = self.cmap(scaled_freqs[word])
            rgb_value_255 = tuple(int(x * 255) for x in rgba_value[:3])
            return rgb_value_255

        wc = WordCloud(width=wcloud_width,
                       height=wcloud_height,
                       prefer_horizontal=0.5,
                       background_color="white",
                       color_func=my_color_function,
                       min_font_size=min_font_size,
                       max_font_size=max_font_size,
                       max_words=20)
        wc.generate_from_frequencies(scaled_freqs)
        ax.axis("off")
        ax.imshow(wc, interpolation="bilinear")

    def plot_world_map_country_count(self, country_counts: pd.Series = None, plot_title: str = None):
        """
        Plot a world map displaying the count of publications per country.

        This method visualizes the count of publications per country on a world map using a color-coded scheme.
        It requires a pandas Series containing the count of publications for each country, indexed by country name.

        Parameters:
        - country_counts (pd.Series, optional): A pandas Series containing the count of publications for each country.
        Defaults to None, in which case the count is automatically obtained using the _get_records_count_per_country method.
        - plot_title (str, optional): Title for the plot. Defaults to None.

        """
        fig = plt.figure(figsize=(7.5, 7.5), constrained_layout=False)
        if country_counts is None:
            country_counts = self._get_records_count_per_country()

        ax_map = fig.add_subplot(projection=ccrs.PlateCarree())

        divider = make_axes_locatable(ax_map)
        ax_colorbar = divider.new_horizontal(size="2.5%", pad=0.1, axes_class=plt.Axes)
        fig.add_axes(ax_colorbar)

        ax_map.set_global()
        ax_map.set_ylim([-60, 80])
        ax_map.add_feature(cfeature.LAND.with_scale('50m'), color='w')
        ax_map.add_feature(cfeature.OCEAN.with_scale('50m'), color='w')
        ax_map.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.5)
        ax_map.add_feature(cfeature.BORDERS.with_scale('50m'), lw=0.6)

        # Get the maximum count for normalization
        max_count = country_counts.max()

        # Colorize Map

        for country_name, count in country_counts.items():
            code = self.country_dict[country_name]
            shpfilename = shpreader.natural_earth(
                resolution='110m',
                category='cultural',
                name='admin_0_countries')
            reader = shpreader.Reader(shpfilename)
            countries_records = reader.records()

            # Normalize count to range [0, 1] for colormap
            norm_count = count / max_count

            facecolor = self.cmap(norm_count)

            for country in countries_records:
                if country.attributes['ADM0_A3'] == code:
                    ax_map.add_geometries(country.geometry, ccrs.PlateCarree(), facecolor=facecolor)

        sm = ScalarMappable(cmap=self.cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])  # Dummy empty array for the ScalarMappable

        # Add colorbar
        tick_positions = np.linspace(0, 1, min(max_count, 5))  # 5 evenly spaced ticks between 0 and 1
        tick_labels = [int(tick_pos * max_count) for tick_pos in tick_positions]  # Corresponding count values
        cbar = plt.colorbar(sm, cax=ax_colorbar, orientation='vertical')
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels)
        cbar.set_label('Total publication count')
        if plot_title:
            ax_map.set_title(plot_title)

    def _get_records_count_per_country(self) -> pd.Series:
        """
        Get the count of records per country from the DataFrame.

        This method calculates the count of records for each country based on the 'affiliation_country' column
        in the DataFrame. It performs necessary adjustments such as standardizing country names before counting.

        Returns:
        - pd.Series: A pandas Series containing the count of records for each country.

        """
        # Perform necessary adjustments to standardize country names
        self.df['affiliation_country'] = self.df['affiliation_country'].str.replace('South Korea', 'Korea, Republic of')
        self.df['affiliation_country'] = self.df['affiliation_country'].str.replace('Iran', 'Iran, Islamic Republic of')
        self.df['affiliation_country'] = self.df['affiliation_country'].str.replace('Taiwan', 'Taiwan, Province of China')

        # Extract the primary country affiliation for each record
        countries = self.df['affiliation_country'].str.split(';').str.get(0)

        # Count the number of records for each country
        return countries.value_counts()
