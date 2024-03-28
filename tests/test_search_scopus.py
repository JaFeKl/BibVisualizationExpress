from BibVisualizationExpress.BibVisualizationExpress import BibVisualizationExpress


def test_search_scopus():
    # Create an object from your library
    bibvis = BibVisualizationExpress()

    if bibvis.scopus is True:
        bibvis.search_scopus("TITLE-ABS-KEY(travel AND time AND model AND as/rs)", False)
        assert len(bibvis.df) >= 150
