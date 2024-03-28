from BibVisualizationExpress.BibVisualizationExpress import BibVisualizationExpress


def test_postprocessing():
    # Create an object from your library
    bibvis = BibVisualizationExpress()

    bibvis.load_csv('example.csv')
    len_before = len(bibvis.df)
    bibvis.remove_empty_doi()
    len_after = len(bibvis.df)
    # we should have removed exactly 20 records without doi
    assert len_before-20 == len_after

    # we should have removed exactly 133 'Articles
    len_before = len_after
    bibvis.remove_record_types(['Article'])
    len_after = len(bibvis.df)
    assert len_before-105 == len_after

    if bibvis.scopus is True:
        len_before = len_after
        bibvis.restrict_languages(['ger'])
        len_after = len(bibvis.df)
        assert len_after == 0
