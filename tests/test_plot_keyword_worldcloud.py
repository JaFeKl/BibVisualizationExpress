from BibVisualizationExpress.BibVisualizationExpress import BibVisualizationExpress


def test_plot_records_over_time():
    # Create an object from your library
    bibvis = BibVisualizationExpress()
    bibvis.load_csv('example.csv')
    bibvis.plot_keyword_worldcloud()
