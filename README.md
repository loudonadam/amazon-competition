# Amazon Competition Prediction

Some of the largest companies in the world are hosts of online marketplaces. Companies like Apple, Microsoft, Amazon, Google, and others. By creating markets within their ecosystems, they are able to flaunt what would normally be considered monopolistic actions due to their control of the platform. Modern regulators are trying to find a way to protect consumers from this behavior, but it is a very interesting topic to me.

I found a dataset on Kaggle (https://www.kaggle.com/datasets/asaniczka/amazon-products-dataset-2023-1-4m-products) which includes 1.4 million amazon products in 248 categories. My goal was to measure which categories Amazon decides to compete in within it's own platform. I take all 1.4 million products and summarize them to the category level while preserving category metrics which describe the prices, volumne, and relevant competition in each market. I then use these metrics to predict whether a category with specific attributes is likely to already or soon have amazon competing within it.

This tool would be useful for three different personas:
First, Amazon itself. Part of the complaints against Amazon is that they have access to a massive amount of data on successful products. They can then analyze that data, introduce their own competing products, and throw them to the top of search results. My tool uses a tiny fraction of what Amazon data scientists have available, but shows how this kind of data could be used to determine what product categories they would to compete in.

Second, other companies. Starting business and engaging that business on Amazon or another platform is a serious risk. Even if you are successful, you may end up facing competition on uneven terms with a behemoth like Amazon. Before entering a new category, using a tool like this may help you understand if you are likely to face competition with the platform owner in the future.

Third, regulatory agencies. Agencies like the FTC or the European Commission may find a tool like this interesting. Understanding what kinds of markets Amazon decides to compete in tells a lot about their motivations and methods of rent extraction. 
