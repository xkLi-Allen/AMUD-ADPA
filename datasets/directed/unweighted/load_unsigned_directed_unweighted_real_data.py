from datasets.directed.unweighted.UsDUw_pygsd import UsDUwPyGSDDataset


def load_unsigned_directed_unweighted_dataset(logger, args, name, root, k, node_split, node_split_id, edge_split,
                                              edge_split_id):
    if name.lower() in ("coraml", "citeseerdir", "chameleondir", "squirreldir", "wikics", "slashdot", "epinions", "wikitalk", \
                        "proteins", "amazon-computers", "amazon-photo", "pubmed", "coauthor-cs", "coauthor-physic", "cornell",\
                        "texas", "wisconsin", "squirrel", "chameleon", "coramlundir", "citeseer", "minesweeper", "tolokers", \
                        "roman", "rating", "question", "squirrelfilterdir", "chameleonfilterdir", "squirrelfilter", "chameleonfilter",\
                        "actor", "cornellundir", "romanundir", "ratingundir", "snap"):
        dataset = UsDUwPyGSDDataset(args, name, root, k, node_split, node_split_id, edge_split, edge_split_id)

    if args.heterogeneity and name.lower() not in ("wikitalk", "slashdot", "epinions"):
        logger.info("Edge homophily: {}, Node homophily:{}, Linkx homophily:{}, adjusted homophily:{}, li homophily:{}".format(round(dataset.edge_homophily, 4),
                                                                                       round(dataset.node_homophily, 4),
                                                                                       round(dataset.linkx_homophily,4),
                                                                                       round(dataset.adjusted_homophily,4),
                                                                                       round(dataset.li_homophily, 4)
                                                                                       ))
    return dataset