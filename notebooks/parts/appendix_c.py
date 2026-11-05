# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Appendix C: the references. C.1 is every work the text cites by author and year, each one resolved to a fetched record before it was written; C.2 is the software, models and data the text names, each with the source the repository itself records (METHOD_REPORT.md section 12, README.md, the pin files in environment/, docs/video/voices.py and the data loading modules of cypmode/)."""

from notebooks.notebook_builder import code, md

# Every entry of C.1 was fetched before it was written: the Crossref record of the DOI for an article, a chapter or a book; the arXiv or bioRxiv abstract
# page for a preprint; the journal, library or archive page for a work that predates the DOI system. The links are those records. The check was made on
# 17 October 2026. The number of entries is counted from the list by the COUNT cell, so that the text's number and the list cannot drift apart.
WORKS = """
1. Abramson, J., Adler, J., Dunger, J., Evans, R., Green, T., Pritzel, A., et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature, 630, 493-500. [doi:10.1038/s41586-024-07487-w](https://doi.org/10.1038/s41586-024-07487-w)
2. Ambroise, C., and McLachlan, G. J. (2002). Selection bias in gene extraction on the basis of microarray gene-expression data. Proceedings of the National Academy of Sciences, 99, 6562-6566. [doi:10.1073/pnas.102102699](https://doi.org/10.1073/pnas.102102699)
3. Bannwarth, C., Ehlert, S., and Grimme, S. (2019). GFN2-xTB─An Accurate and Broadly Parametrized Self-Consistent Tight-Binding Quantum Chemical Method with Multipole Electrostatics and Density-Dependent Dispersion Contributions. Journal of Chemical Theory and Computation, 15, 1652-1671. [doi:10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176)
4. Becke, A. D. (1993). Density-functional thermochemistry. III. The role of exact exchange. The Journal of Chemical Physics, 98, 5648-5652. [doi:10.1063/1.464913](https://doi.org/10.1063/1.464913)
5. Bemis, G. W., and Murcko, M. A. (1996). The Properties of Known Drugs. 1. Molecular Frameworks. Journal of Medicinal Chemistry, 39, 2887-2893. [doi:10.1021/jm9602928](https://doi.org/10.1021/jm9602928)
6. Biasini, M., Schmidt, T., Bienert, S., Mariani, V., Studer, G., Haas, J., et al. (2013). OpenStructure: an integrated software framework for computational structural biology. Acta Crystallographica Section D Biological Crystallography, 69, 701-709. [doi:10.1107/s0907444913007051](https://doi.org/10.1107/s0907444913007051)
7. Blanksby, S. J., and Ellison, G. B. (2003). Bond Dissociation Energies of Organic Molecules. Accounts of Chemical Research, 36, 255-263. [doi:10.1021/ar020230d](https://doi.org/10.1021/ar020230d)
8. Bondi, A. (1964). van der Waals Volumes and Radii. The Journal of Physical Chemistry, 68, 441-451. [doi:10.1021/j100785a001](https://doi.org/10.1021/j100785a001)
9. Born, M., and Oppenheimer, R. (1927). Zur Quantentheorie der Molekeln. Annalen der Physik, 389, 457-484. [doi:10.1002/andp.19273892002](https://doi.org/10.1002/andp.19273892002)
10. Breiman, L. (1996). Bagging Predictors. Machine Learning, 24, 123-140. [doi:10.1023/a:1018054314350](https://doi.org/10.1023/a:1018054314350)
11. Breiman, L. (2001). Random Forests. Machine Learning, 45, 5-32. [doi:10.1023/a:1010933404324](https://doi.org/10.1023/a:1010933404324)
12. Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. Monthly Weather Review, 78, 1–3. [`10.1175/1520-0493(1950)078<0001:vofeit>2.0.co;2`](https://doi.org/10.1175/1520-0493%281950%29078%3C0001%3Avofeit%3E2.0.co%3B2)
13. Briggs, G. E., and Haldane, J. B. S. (1925). A Note on the Kinetics of Enzyme Action. Biochemical Journal, 19, 338-339. [doi:10.1042/bj0190338](https://doi.org/10.1042/bj0190338)
14. Butina, D. (1999). Unsupervised Data Base Clustering Based on Daylight's Fingerprint and Tanimoto Similarity: A Fast and Automated Way To Cluster Small and Large Data Sets. Journal of Chemical Information and Computer Sciences, 39, 747-750. [doi:10.1021/ci9803381](https://doi.org/10.1021/ci9803381)
15. Buttenschoen, M., Morris, G. M., and Deane, C. M. (2024). PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences. Chemical Science, 15, 3130-3139. [doi:10.1039/d3sc04185a](https://doi.org/10.1039/d3sc04185a)
16. Carhart, R. E., Smith, D. H., and Venkataraghavan, R. (1985). Atom pairs as molecular features in structure-activity studies: definition and applications. Journal of Chemical Information and Computer Sciences, 25, 64-73. [doi:10.1021/ci00046a002](https://doi.org/10.1021/ci00046a002)
17. Chen, T., and Guestrin, C. (2016). XGBoost: a scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785–794. [doi:10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
18. Cheng, Y.-C., and Prusoff, W. H. (1973). Relationship between the inhibition constant (K_I) and the concentration of inhibitor which causes 50 per cent inhibition (I_50) of an enzymatic reaction. Biochemical Pharmacology, 22, 3099–3108. [doi:10.1016/0006-2952(73)90196-2](https://doi.org/10.1016/0006-2952(73)90196-2)
19. Chithrananda, S., Grand, G., and Ramsundar, B. (2020). ChemBERTa: large-scale self-supervised pretraining for molecular property prediction. arXiv preprint arXiv:2010.09885. [arXiv:2010.09885](https://doi.org/10.48550/arXiv.2010.09885)
20. Clayden, J., Greeves, N., and Warren, S. (2012). Organic Chemistry (2nd ed.). Oxford University Press, Oxford. [doi:10.1093/hesc/9780199270293.001.0001](https://doi.org/10.1093/hesc/9780199270293.001.0001)
21. Correia, M. A., and Ortiz de Montellano, P. R. (2005). Inhibition of cytochrome P450 enzymes. In P. R. Ortiz de Montellano (Ed.), Cytochrome P450: structure, mechanism, and biochemistry (3rd ed., pp. 247–322). Springer. [doi:10.1007/0-387-27447-2_7](https://doi.org/10.1007/0-387-27447-2_7)
22. Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. Mathematics of Control, Signals, and Systems, 2, 303-314. [doi:10.1007/bf02551274](https://doi.org/10.1007/bf02551274)
23. de Groot, M. J. (2006). Designing better drugs: predicting cytochrome P450 metabolism. Drug Discovery Today, 11, 601-606. [doi:10.1016/j.drudis.2006.05.001](https://doi.org/10.1016/j.drudis.2006.05.001)
24. Durant, J. L., Leland, B. A., Henry, D. R., and Nourse, J. G. (2002). Reoptimization of MDL Keys for Use in Drug Discovery. Journal of Chemical Information and Computer Sciences, 42, 1273-1280. [doi:10.1021/ci010132r](https://doi.org/10.1021/ci010132r)
25. Ehlert, S., Stahn, M., Spicher, S., and Grimme, S. (2021). Robust and Efficient Implicit Solvation Model for Fast Semiempirical Methods. Journal of Chemical Theory and Computation, 17, 4250-4261. [doi:10.1021/acs.jctc.1c00471](https://doi.org/10.1021/acs.jctc.1c00471)
26. Ertl, P., Rohde, B., and Selzer, P. (2000). Fast Calculation of Molecular Polar Surface Area as a Sum of Fragment-Based Contributions and Its Application to the Prediction of Drug Transport Properties. Journal of Medicinal Chemistry, 43, 3714-3717. [doi:10.1021/jm000942e](https://doi.org/10.1021/jm000942e)
27. Fisher, R. A. (1935). The design of experiments. Oliver and Boyd. [OpenLibrary OL1153859W](https://openlibrary.org/works/OL1153859W)
28. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 29, 1189-1232. [doi:10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
29. Gal, Y., and Ghahramani, Z. (2016). Dropout as a Bayesian approximation: representing model uncertainty in deep learning. arXiv preprint arXiv:1506.02142 (ICML 2016). [arXiv:1506.02142](https://doi.org/10.48550/arXiv.1506.02142)
30. Gedeck, P., Rohde, B., and Bartels, C. (2006). QSAR − How Good Is It in Practice? Comparison of Descriptor Sets on an Unbiased Cross Section of Corporate Data Sets. Journal of Chemical Information and Modeling, 46, 1924-1936. [doi:10.1021/ci050413p](https://doi.org/10.1021/ci050413p)
31. Geurts, P., Ernst, D., and Wehenkel, L. (2006). Extremely randomized trees. Machine Learning, 63, 3-42. [doi:10.1007/s10994-006-6226-1](https://doi.org/10.1007/s10994-006-6226-1)
32. Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., and Dahl, G. E. (2017). Neural message passing for quantum chemistry. arXiv preprint arXiv:1704.01212 (ICML 2017). [arXiv:1704.01212](https://doi.org/10.48550/arXiv.1704.01212)
33. Guerra, J. V. d. S., Ribeiro-Filho, H. V., Jara, G. E., Bortot, L. O., Pereira, J. G. d. C., and Lopes-de-Oliveira, P. S. (2021). pyKVFinder: an efficient and integrable Python package for biomolecular cavity detection and characterization in data science. BMC Bioinformatics, 22, 607. [doi:10.1186/s12859-021-04519-4](https://doi.org/10.1186/s12859-021-04519-4)
34. Halgren, T. A. (1996). Merck molecular force field. I. Basis, form, scope, parameterization, and performance of MMFF94. Journal of Computational Chemistry, 17, 490–519. [`10.1002/(sici)1096-987x(199604)17:5/6<490::aid-jcc1>3.0.co;2-p`](https://doi.org/10.1002/%28sici%291096-987x%28199604%2917%3A5/6%3C490%3A%3Aaid-jcc1%3E3.0.co%3B2-p)
35. Hall, G. G. (1951). The molecular orbital theory of chemical valency VIII. A method of calculating ionization potentials. Proceedings of the Royal Society of London. Series A. Mathematical and Physical Sciences, 205, 541-552. [doi:10.1098/rspa.1951.0048](https://doi.org/10.1098/rspa.1951.0048)
36. Hanley, J. A., and McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. Radiology, 143, 29-36. [doi:10.1148/radiology.143.1.7063747](https://doi.org/10.1148/radiology.143.1.7063747)
37. Heckman, J. J. (1979). Sample Selection Bias as a Specification Error. Econometrica, 47, 153. [doi:10.2307/1912352](https://doi.org/10.2307/1912352)
38. Hill, A. V. (1910). The possible effects of the aggregation of the molecules of haemoglobin on its dissociation curves. The Journal of Physiology, 40 (Suppl.), iv–vii. [doi:10.1113/jphysiol.1910.sp001386](https://doi.org/10.1113/jphysiol.1910.sp001386)
39. Hoerl, A. E., and Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. Technometrics, 12, 55-67. [doi:10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
40. Hohenberg, P., and Kohn, W. (1964). Inhomogeneous Electron Gas. Physical Review, 136, B864-B871. [doi:10.1103/physrev.136.b864](https://doi.org/10.1103/physrev.136.b864)
41. Hollmann, N., Müller, S., Eggensperger, K., and Hutter, F. (2022). TabPFN: a transformer that solves small tabular classification problems in a second. arXiv preprint arXiv:2207.01848. [arXiv:2207.01848](https://doi.org/10.48550/arXiv.2207.01848)
42. Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6, 65–70. [jstor.org/stable/4615733](https://www.jstor.org/stable/4615733)
43. Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. Neural Networks, 4, 251-257. [doi:10.1016/0893-6080(91)90009-t](https://doi.org/10.1016/0893-6080(91)90009-t)
44. Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components. Journal of Educational Psychology, 24, 417–441 and 498–520 (in two parts). [doi:10.1037/h0071325](https://doi.org/10.1037/h0071325), [doi:10.1037/h0070888](https://doi.org/10.1037/h0070888)
45. Hückel, E. (1931). Quantentheoretische Beiträge zum Benzolproblem. Zeitschrift für Physik, 70, 204–286. [doi:10.1007/bf01339530](https://doi.org/10.1007/bf01339530)
46. Hussain, J., and Rea, C. (2010). Computationally Efficient Algorithm to Identify Matched Molecular Pairs (MMPs) in Large Data Sets. Journal of Chemical Information and Modeling, 50, 339-348. [doi:10.1021/ci900450m](https://doi.org/10.1021/ci900450m)
47. Jaccard, P. (1901). Distribution de la flore alpine dans le bassin des Dranses et dans quelques régions voisines. Bulletin de la Société Vaudoise des Sciences Naturelles, 37, 241–272. [doi:10.5169/seals-266440](https://doi.org/10.5169/seals-266440)
48. Janežič, M., Valjavec, K., Bergant Loboda, K., Herlah, B., Ogris, I., Kozorog, M., et al. (2021). Dynophore-Based Approach in Virtual Screening: A Case of Human DNA Topoisomerase IIα. International Journal of Molecular Sciences, 22, 13474. [doi:10.3390/ijms222413474](https://doi.org/10.3390/ijms222413474)
49. Johnson, M. A., and Maggiora, G. M. (Eds.) (1990). Concepts and applications of molecular similarity. Wiley. [OpenLibrary OL19334471W](https://openlibrary.org/works/OL19334471W)
50. Jones, D. R., Schonlau, M., and Welch, W. J. (1998). Efficient Global Optimization of Expensive Black-Box Functions. Journal of Global Optimization, 13, 455-492. [doi:10.1023/a:1008306431147](https://doi.org/10.1023/a:1008306431147)
51. Jumper, J., Evans, R., Pritzel, A., Green, T., Figurnov, M., Ronneberger, O., et al. (2021). Highly accurate protein structure prediction with AlphaFold. Nature, 596, 583-589. [doi:10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)
52. Kalgutkar, A. S., Obach, R. S., and Maurer, T. S. (2007). Mechanism-based inactivation of cytochrome P450 enzymes: chemical mechanisms, structure–activity relationships and relationship to clinical drug–drug interactions and idiosyncratic adverse drug reactions. Current Drug Metabolism, 8, 407–447. [doi:10.2174/138920007780866807](https://doi.org/10.2174/138920007780866807)
53. Karras, T., Aittala, M., Aila, T., and Laine, S. (2022). Elucidating the Design Space of Diffusion-Based Generative Models. In Advances in Neural Information Processing Systems 35, pp. 26565-26577. [doi:10.52202/068431-1926](https://doi.org/10.52202/068431-1926)
54. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. In Advances in Neural Information Processing Systems 30 (NIPS 2017), pp. 3146–3154. [proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
55. Kekulé, A. (1865). Sur la constitution des substances aromatiques. Bulletin de la Société Chimique de Paris, 3, 98–110. [Gallica, BnF](http://gallica.bnf.fr/ark:/12148/bpt6k281952v/f102.image)
56. Kenny, P. W., and Sadowski, J. (2005). Structure modification in chemical databases. In Chemoinformatics in drug discovery (Methods and Principles in Medicinal Chemistry, pp. 271–285). Wiley-VCH. [doi:10.1002/3527603743.ch11](https://doi.org/10.1002/3527603743.ch11)
57. Kingma, D. P., and Ba, J. (2015). Adam: a method for stochastic optimization. arXiv preprint arXiv:1412.6980 (ICLR 2015). [arXiv:1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
58. Kohn, W., and Sham, L. J. (1965). Self-Consistent Equations Including Exchange and Correlation Effects. Physical Review, 140, A1133-A1138. [doi:10.1103/physrev.140.a1133](https://doi.org/10.1103/physrev.140.a1133)
59. Leach, A. R. (2001). Molecular modelling: principles and applications (2nd ed.). Prentice Hall. [OpenLibrary OL8402650W](https://openlibrary.org/works/OL8402650W)
60. Leach, A. G. (2014). Tactics to avoid inhibition of cytochrome P450s. In N. A. Meanwell (Ed.), Tactics in contemporary drug design (Topics in Medicinal Chemistry, pp. 107–158). Springer. [doi:10.1007/7355_2013_25](https://doi.org/10.1007/7355_2013_25)
61. Lee, C., Yang, W., and Parr, R. G. (1988). Development of the Colle-Salvetti correlation-energy formula into a functional of the electron density. Physical Review B, 37, 785-789. [doi:10.1103/physrevb.37.785](https://doi.org/10.1103/physrevb.37.785)
62. Lim, H.-K., Duczak, N., Brougham, L., Elliot, M., Patel, K., and Chan, K. (2005). Automated screening with confirmation of mechanism-based inactivation of CYP3A4, CYP2C9, CYP2C19, CYP2D6, and CYP1A2 in pooled human liver microsomes. Drug Metabolism and Disposition, 33, 1211–1219. [doi:10.1124/dmd.104.003475](https://doi.org/10.1124/dmd.104.003475)
63. Lipinski, C. A., Lombardo, F., Dominy, B. W., and Feeney, P. J. (1997). Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings. Advanced Drug Delivery Reviews, 23, 3-25. [doi:10.1016/s0169-409x(96)00423-1](https://doi.org/10.1016/s0169-409x(96)00423-1)
64. Lipkus, A. H. (1999). A proof of the triangle inequality for the Tanimoto distance. Journal of Mathematical Chemistry, 26, 263-265. [doi:10.1023/a:1019154432472](https://doi.org/10.1023/a:1019154432472)
65. Loshchilov, I., and Hutter, F. (2019). Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (ICLR 2019). [arXiv:1711.05101](https://doi.org/10.48550/arXiv.1711.05101)
66. Mann, H. B., and Whitney, D. R. (1947). On a Test of Whether one of Two Random Variables is Stochastically Larger than the Other. The Annals of Mathematical Statistics, 18, 50-60. [doi:10.1214/aoms/1177730491](https://doi.org/10.1214/aoms/1177730491)
67. Mariani, V., Biasini, M., Barbato, A., and Schwede, T. (2013). lDDT: a local superposition-free score for comparing protein structures and models using distance difference tests. Bioinformatics, 29, 2722-2728. [doi:10.1093/bioinformatics/btt473](https://doi.org/10.1093/bioinformatics/btt473)
68. Matthews, B. (1975). Comparison of the predicted and observed secondary structure of T4 phage lysozyme. Biochimica et Biophysica Acta (BBA) - Protein Structure, 405, 442-451. [doi:10.1016/0005-2795(75)90109-9](https://doi.org/10.1016/0005-2795(75)90109-9)
69. Michaelis, L., and Menten, M. L. (1913). Die Kinetik der Invertinwirkung. Biochemische Zeitschrift, 49, 333–369; English translation in Biochemistry, 50, 8264–8269 (2011). [doi:10.1021/bi201284u](https://doi.org/10.1021/bi201284u)
70. Mirdita, M., Schütze, K., Moriwaki, Y., Heo, L., Ovchinnikov, S., and Steinegger, M. (2022). ColabFold: making protein folding accessible to all. Nature Methods, 19, 679-682. [doi:10.1038/s41592-022-01488-1](https://doi.org/10.1038/s41592-022-01488-1)
71. Mobley, D. L., and Guthrie, J. P. (2014). FreeSolv: a database of experimental and calculated hydration free energies, with input files. Journal of Computer-Aided Molecular Design, 28, 711-720. [doi:10.1007/s10822-014-9747-x](https://doi.org/10.1007/s10822-014-9747-x)
72. Morgan, H. L. (1965). The Generation of a Unique Machine Description for Chemical Structures-A Technique Developed at Chemical Abstracts Service. Journal of Chemical Documentation, 5, 107-113. [doi:10.1021/c160017a018](https://doi.org/10.1021/c160017a018)
73. Nadeau, C., and Bengio, Y. (2003). Inference for the Generalization Error. Machine Learning, 52, 239-281. [doi:10.1023/a:1024068626366](https://doi.org/10.1023/a:1024068626366)
74. Nilakantan, R., Bauman, N., Dixon, J. S., and Venkataraghavan, R. (1987). Topological torsion: a new molecular descriptor for SAR applications. Comparison with other descriptors. Journal of Chemical Information and Computer Sciences, 27, 82-85. [doi:10.1021/ci00054a008](https://doi.org/10.1021/ci00054a008)
75. Oliveira, S. H., Ferraz, F. A., Honorato, R. V., Xavier-Neto, J., Sobreira, T. J., and de Oliveira, P. S. (2014). KVFinder: steered identification of protein cavities as a PyMOL plugin. BMC Bioinformatics, 15, 197. [doi:10.1186/1471-2105-15-197](https://doi.org/10.1186/1471-2105-15-197)
76. Omura, T., and Sato, R. (1962). A New Cytochrome in Liver Microsomes. Journal of Biological Chemistry, 237, PC1375-PC1376. [doi:10.1016/s0021-9258(18)60338-2](https://doi.org/10.1016/s0021-9258(18)60338-2)
77. Omura, T., and Sato, R. (1964). The Carbon Monoxide-binding Pigment of Liver Microsomes. Journal of Biological Chemistry, 239, 2370-2378. [doi:10.1016/s0021-9258(20)82244-3](https://doi.org/10.1016/s0021-9258(20)82244-3)
78. Orr, S. T. M., Ripp, S. L., Ballard, T. E., Henderson, J. L., Scott, D. O., Obach, R. S., et al. (2012). Mechanism-Based Inactivation (MBI) of Cytochrome P450 Enzymes: Structure–Activity Relationships and Discovery Strategies To Mitigate Drug–Drug Interaction Risks. Journal of Medicinal Chemistry, 55, 4896-4933. [doi:10.1021/jm300065h](https://doi.org/10.1021/jm300065h)
79. Orta, A. K., Schäfer, J.-H., Correy, G. J., Norman, J. O., Pampel, J., Huddleston, K. K., et al. (2026). Structural plasticity and ligand promiscuity of CYP3A4 revealed by cryo-EM. bioRxiv preprint (W. P. Walters is a co-author; the text cites this preprint as Walters and co-workers). [doi:10.64898/2026.09.01.748687](https://doi.org/10.64898/2026.09.01.748687)
80. Parr, R. G., and Yang, W. (1984). Density functional approach to the frontier-electron theory of chemical reactivity. Journal of the American Chemical Society, 106, 4049-4050. [doi:10.1021/ja00326a036](https://doi.org/10.1021/ja00326a036)
81. Passaro, S., Corso, G., Wohlwend, J., Reveiz, M., Thaler, S., Somnath, V. R., et al. (2025). Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction. bioRxiv preprint. [doi:10.1101/2025.06.14.659707](https://doi.org/10.1101/2025.06.14.659707)
82. Pearson, K. (1901). LIII. On lines and planes of closest fit to systems of points in space. The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science, 2, 559-572. [doi:10.1080/14786440109462720](https://doi.org/10.1080/14786440109462720)
83. Perdew, J. P., Burke, K., and Ernzerhof, M. (1996). Generalized Gradient Approximation Made Simple. Physical Review Letters, 77, 3865-3868. [doi:10.1103/physrevlett.77.3865](https://doi.org/10.1103/physrevlett.77.3865)
84. Platt, J. C. (1999). Probabilities for SV machines. In Advances in Large-Margin Classifiers, pp. 61–74. MIT Press (printed 2000). [doi:10.7551/mitpress/1113.003.0008](https://doi.org/10.7551/mitpress/1113.003.0008)
85. Poulos, T. L., and Johnson, E. F. (2005). Structures of cytochrome P450 enzymes. In P. R. Ortiz de Montellano (Ed.), Cytochrome P450: structure, mechanism, and biochemistry (3rd ed., pp. 87–114). Springer. [doi:10.1007/0-387-27447-2_3](https://doi.org/10.1007/0-387-27447-2_3)
86. Qu, J., Holzmüller, D., Varoquaux, G., and Le Morvan, M. (2025). TabICL: a tabular foundation model for in-context learning on large data. arXiv preprint arXiv:2502.05564 (ICML 2025). [arXiv:2502.05564](https://doi.org/10.48550/arXiv.2502.05564)
87. Ralaivola, L., Swamidass, S. J., Saigo, H., and Baldi, P. (2005). Graph kernels for chemical informatics. Neural Networks, 18, 1093-1110. [doi:10.1016/j.neunet.2005.07.009](https://doi.org/10.1016/j.neunet.2005.07.009)
88. Riniker, S., and Landrum, G. A. (2015). Better Informed Distance Geometry: Using What We Know To Improve Conformation Generation. Journal of Chemical Information and Modeling, 55, 2562-2574. [doi:10.1021/acs.jcim.5b00654](https://doi.org/10.1021/acs.jcim.5b00654)
89. Rogers, D., and Hahn, M. (2010). Extended-Connectivity Fingerprints. Journal of Chemical Information and Modeling, 50, 742-754. [doi:10.1021/ci100050t](https://doi.org/10.1021/ci100050t)
90. Roothaan, C. C. J. (1951). New Developments in Molecular Orbital Theory. Reviews of Modern Physics, 23, 69-89. [doi:10.1103/revmodphys.23.69](https://doi.org/10.1103/revmodphys.23.69)
91. Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. Psychological Review, 65, 386-408. [doi:10.1037/h0042519](https://doi.org/10.1037/h0042519)
92. Rowland, P., Blaney, F. E., Smyth, M. G., Jones, J. J., Leydon, V. R., Oxbrow, A. K., et al. (2006). Crystal Structure of Human Cytochrome P450 2D6. Journal of Biological Chemistry, 281, 7614-7622. [doi:10.1074/jbc.M511232200](https://doi.org/10.1074/jbc.M511232200)
93. Rumelhart, D. E., Hinton, G. E., and Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323, 533-536. [doi:10.1038/323533a0](https://doi.org/10.1038/323533a0)
94. Schenkman, J. B., Remmer, H., and Estabrook, R. W. (1967). Spectral studies of drug interaction with hepatic microsomal cytochrome. Molecular Pharmacology, 3, 113–123. [doi:10.1016/s0026-895x(25)14748-2](https://doi.org/10.1016/s0026-895x(25)14748-2)
95. Seung, H. S., Opper, M., and Sompolinsky, H. (1992). Query by committee. In Proceedings of the fifth annual workshop on Computational learning theory, pp. 287-294. [doi:10.1145/130385.130417](https://doi.org/10.1145/130385.130417)
96. Sevrioukova, I. F. (2019). Structural Insights into the Interaction of Cytochrome P450 3A4 with Suicide Substrates: Mibefradil, Azamulin and 6′,7′-Dihydroxybergamottin. International Journal of Molecular Sciences, 20, 4245. [doi:10.3390/ijms20174245](https://doi.org/10.3390/ijms20174245)
97. Shrake, A., and Rupley, J. (1973). Environment and exposure to solvent of protein atoms. Lysozyme and insulin. Journal of Molecular Biology, 79, 351-371. [doi:10.1016/0022-2836(73)90011-9](https://doi.org/10.1016/0022-2836(73)90011-9)
98. Silverman, R. B. (1995). Mechanism-based enzyme inactivators. In Methods in Enzymology, pp. 240–283. Elsevier. [doi:10.1016/0076-6879(95)49038-8](https://doi.org/10.1016/0076-6879(95)49038-8)
99. Stewart, J. J. P. (2007). Optimization of parameters for semiempirical methods V: Modification of NDDO approximations and application to 70 elements. Journal of Molecular Modeling, 13, 1173-1213. [doi:10.1007/s00894-007-0233-4](https://doi.org/10.1007/s00894-007-0233-4)
100. Stiefl, N., Watson, I. A., Baumann, K., and Zaliani, A. (2006). ErG: 2D pharmacophore descriptions for scaffold hopping. Journal of Chemical Information and Modeling, 46, 208–220. [doi:10.1021/ci050457y](https://doi.org/10.1021/ci050457y)
101. Swamidass, S. J., Chen, J., Bruand, J., Phung, P., Ralaivola, L., and Baldi, P. (2005). Kernels for small molecules and the prediction of mutagenicity, toxicity and anti-cancer activity. Bioinformatics, 21, i359-i368. [doi:10.1093/bioinformatics/bti1055](https://doi.org/10.1093/bioinformatics/bti1055)
102. Szabo, A., and Ostlund, N. S. (1989). Modern quantum chemistry: introduction to advanced electronic structure theory. McGraw-Hill (1989 edition; Dover reprint). [OpenLibrary OL3254911W](https://openlibrary.org/works/OL3254911W)
103. Tanimoto, T. T. (1958). An elementary mathematical theory of classification and prediction. IBM technical report; NTIS PB167360, 12 pp. [ntrl.ntis.gov](https://ntrl.ntis.gov/NTRL/dashboard/searchResults/titleDetail/PB167360.xhtml)
104. Tibshirani, R. (1996). Regression Shrinkage and Selection Via the Lasso. Journal of the Royal Statistical Society Series B: Statistical Methodology, 58, 267-288. [doi:10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
105. Tukey, J. W. (1949). Comparing Individual Means in the Analysis of Variance. Biometrics, 5, 99. [doi:10.2307/3001913](https://doi.org/10.2307/3001913)
106. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., et al. (2017). Attention is all you need. arXiv preprint arXiv:1706.03762 (NeurIPS 2017). [arXiv:1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
107. Veith, H., Southall, N., Huang, R., James, T., Fayne, D., Artemenko, N., et al. (2009). Comprehensive characterization of cytochrome P450 isozyme selectivity across chemical libraries. Nature Biotechnology, 27, 1050-1055. [doi:10.1038/nbt.1581](https://doi.org/10.1038/nbt.1581)
108. Weininger, D. (1988). SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules. Journal of Chemical Information and Computer Sciences, 28, 31-36. [doi:10.1021/ci00057a005](https://doi.org/10.1021/ci00057a005)
109. Weininger, D., Weininger, A., and Weininger, J. L. (1989). SMILES. 2. Algorithm for generation of unique SMILES notation. Journal of Chemical Information and Computer Sciences, 29, 97-101. [doi:10.1021/ci00062a008](https://doi.org/10.1021/ci00062a008)
110. Williams, P. A., Cosme, J., Vinković, D. M., Ward, A., Angove, H. C., Day, P. J., et al. (2004). Crystal Structures of Human Cytochrome P450 3A4 Bound to Metyrapone and Progesterone. Science, 305, 683-686. [doi:10.1126/science.1099736](https://doi.org/10.1126/science.1099736)
111. Wohlwend, J., Corso, G., Passaro, S., Getz, N., Reveiz, M., Leidal, K., et al. (2024). Boltz-1 Democratizing Biomolecular Interaction Modeling. bioRxiv preprint. [doi:10.1101/2024.11.19.624167](https://doi.org/10.1101/2024.11.19.624167)
112. Wolpert, D. H. (1992). Stacked generalization. Neural Networks, 5, 241-259. [doi:10.1016/s0893-6080(05)80023-1](https://doi.org/10.1016/s0893-6080(05)80023-1)
113. Woolf, B. (1955). On estimating the relation between blood group and disease. Annals of Human Genetics, 19, 251–253. [doi:10.1111/j.1469-1809.1955.tb01348.x](https://doi.org/10.1111/j.1469-1809.1955.tb01348.x)
114. Yang, K., Swanson, K., Jin, W., Coley, C., Eiden, P., Gao, H., et al. (2019). Analyzing Learned Molecular Representations for Property Prediction. Journal of Chemical Information and Modeling, 59, 3370-3388. [doi:10.1021/acs.jcim.9b00237](https://doi.org/10.1021/acs.jcim.9b00237)
115. Zadrozny, B., and Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. In Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 694-699. [doi:10.1145/775047.775151](https://doi.org/10.1145/775047.775151)
"""

COUNT = r'''# The number of entries of C.1, counted from the list itself so that a sentence of the text cannot disagree with the list it describes.
import re

from notebooks.parts.appendix_c import WORKS

n_works = len([line for line in WORKS.splitlines() if re.match(r"^\d+\. ", line)])
'''

VERSIONS = r'''# The programs named above, with the version each one is pinned at and the pin file it was read from. A version is never typed into the text: it is read from the file in
# environment/ that pins it (name==version in the pip files, name=version in the conda files), so that a table of versions cannot describe an environment other than the one beside it.
import re


def pinned_versions():
    found = {}
    for file_name in ("requirements-lock.txt", "requirements-lock-ml.txt", "openmm-environment.yml", "ost-environment.yml"):
        for line in (ROOT / "environment" / file_name).read_text().splitlines():
            line = line.strip()
            match = re.match(r"^([A-Za-z0-9_.+-]+)==([^;#\s]+)", line) or re.match(r"^-\s+([A-Za-z0-9_.+-]+)=([^\s=#]+)$", line)
            if match:
                found[(file_name, match.group(1).lower())] = match.group(2)
    return found

pins = pinned_versions()
named_programs = [
    ("RDKit", "rdkit", "requirements-lock.txt"), ("LightGBM", "lightgbm", "requirements-lock.txt"),
    ("scikit-learn", "scikit-learn", "requirements-lock.txt"), ("Chemprop", "chemprop", "requirements-lock-ml.txt"),
    ("PyTorch", "torch", "requirements-lock-ml.txt"), ("tblite", "tblite", "requirements-lock.txt"),
    ("OpenMM", "openmm", "openmm-environment.yml"), ("OpenStructure", "openstructure", "ost-environment.yml"),
    ("PoseBusters", "posebusters", "requirements-lock.txt"), ("TabICL", "tabicl", "requirements-lock.txt"),
    ("pandas", "pandas", "requirements-lock.txt"), ("NumPy", "numpy", "requirements-lock.txt"),
]
program_table = pd.DataFrame([{"program": name, "version": pins.get((file_name, package), "not in this pin file"), "pinned in": "environment/" + file_name} for name, package, file_name in named_programs])
show_table(program_table, floatfmt=".0f")
n_programs = len(program_table)
n_lock_entries = len([line for line in (ROOT / "environment" / "requirements-lock.txt").read_text().splitlines() if "==" in line])
'''

LICENCE_TERMS = r'''# The terms of the sources and models behind the data record, read from the policy of the record (data_record/record.json), where each licence is written as it was read.
import json

record_policy = json.loads((ROOT / "data_record/record.json").read_text())
terms_table = pd.DataFrame(record_policy["terms_read"])
show_table(terms_table, floatfmt=".0f")
n_terms = len(terms_table)
n_terms_left_out = int((terms_table["in the record"] == "not in the record").sum())
'''

CELLS = [
    md("""# Appendix C. References

This appendix gives the origin of every method, formula and constant that the notebook did not invent, and the record of the software, the models and the data that it runs on. It is a list for attribution, not for understanding: the parts of the notebook teach each idea in full, and the reader who wants the source of a formula has it here.

Every entry of C.1 was resolved against a record before it was written down. For an article, a chapter, a proceedings paper or a book with a DOI, the record is the Crossref entry of that DOI, and its authors, title, venue, volume, pages and year are the record's. For a preprint (the Transformer paper, the message-passing paper, ChemBERTa, TabPFN, TabICL, Adam, AdamW, dropout as a Bayesian approximation and the two Boltz preprints) the record is the arXiv or bioRxiv abstract page, whose DOI is given. For the works that predate the DOI system the record is the journal, archive or catalogue page that holds the work: the digitised article for Kekulé (1865) and Jaccard (1901), the proceedings volume for Hill (1910), the library catalogue for Fisher (1935) and for the other three books (Johnson and Maggiora, 1990; Szabo and Ostlund, 1989; Leach, 2001), the report number for Tanimoto (1958), and the English translation for Michaelis and Menten (1913). The check was made on 17 October 2026, and the link at the end of an entry is the record that was read.

The entries are in alphabetical order of the first author's surname, and a work of two or more authors is under its first author. The number in front of an entry is only a count. The year is the year the text cites; where a conference paper or a book chapter was printed in a later year (Platt, 1999; Leach and co-workers, 2014) the entry says so and gives the record's year. Two names of the text are kept as the text writes them (`de Groot`, `Ortiz de Montellano`), and the preprint the text cites as `Walters and co-workers, 2026` is entered under its first author, `Orta`, with the reason in the entry.

## C.1 Works cited

> **Field:** Mathematics · Statistics · Biology · Chemistry · Cheminformatics · Machine learning · Structural biology · Software engineering

Below is every author-and-year citation of the text, and nothing else: a work the text does not cite is not in the list, and a work it does cite appears exactly once. An entry whose venue is a book chapter or a proceedings paper gives the book or the proceedings, its pages and its publisher; an entry whose work is a preprint says so; the link opens the record that was checked.
"""),
    md(WORKS),
    code(COUNT),
    md("""There are {{=n_works}} entries. Each one is a work the text cites by author and year; the few works the text names without a year (Kenny and Sadowski; Hollmann and co-workers) are here as well, because the text asks the reader to consult them."""),
    md("""## C.2 Software, models and data

> **Field:** Software engineering · Data science

The notebook runs on programs and data that others wrote and measured, and those have a source too. This section lists them from the place where the repository itself records the source: `METHOD_REPORT.md` section 12 (the credits and the prior art), the References section of `README.md` (the links), the pin files in `environment/` (the versions), the docstring of `docs/video/voices.py` (the voice of the film) and the modules of `cypmode/` that load a data set. A version is never typed into this appendix: the next cell reads each one from the file that pins it, and a pin that changes changes the table.
"""),
    code(VERSIONS),
    md("""The table holds {{=n_programs}} programs, each with the pin file whose version it shows; the main lock file alone pins {{=n_lock_entries}} entries.

**Programs.** **RDKit** supplies the descriptors, the fingerprints and the SMARTS matching, and `README.md` records its home as https://www.rdkit.org. **LightGBM** is the gradient-boosted model of most of the notebook: `README.md` credits it and the work of Ke and co-workers (2017). **scikit-learn** supplies the calibration, the isotonic regression and the metrics. **Chemprop** is the message-passing network of the neural models, credited in `README.md` to Yang and co-workers (2019). **PyTorch** runs the neural models. **tblite** implements the GFN2-xTB method of the physics block. **OpenMM** and `openmmforcefields` run the force-field relaxation of a pose, and **OpenStructure** the structural scoring; these two live in conda environments of their own, `environment/openmm-environment.yml` and `environment/ost-environment.yml`. **PoseBusters** is the set of pose checks of the structure track, and **TabICL** the in-context tabular model, each from its pin file; **pandas** and **NumPy** are the tabular and numerical base of every cell.

**Models and weights.** **Boltz-2** is the co-folding model of the structure track; `README.md` records it at https://github.com/jwohlwend/boltz under the MIT licence, and its paper is Passaro and co-workers (2025). **OpenFold3** is the OpenFold consortium's reimplementation of AlphaFold 3; `METHOD_REPORT.md` section 12 names it and the package is installed in `.venv-openfold3`. **BioEmu2** is the all-atom diffusion model that was examined as a third structure engine; the docstring of `cypmode/validation/bioemu2.py` records it as the model of Microsoft Research under the MIT licence, at https://github.com/microsoft/bioemu2, with its public checkpoint at https://huggingface.co/microsoft/bioemu2. **CheMeleon** supplies the frozen embeddings, and `METHOD_REPORT.md` section 12 records the weights at Zenodo, [doi:10.5281/zenodo.15426600](https://doi.org/10.5281/zenodo.15426600). **ChemBERTa** is the checkpoint `seyonec/ChemBERTa-zinc-base-v1` on Hugging Face, from Chithrananda and co-workers (2020), and **TabPFN** is the in-context tabular model of Hollmann and co-workers (2022). **Kokoro-82M** speaks the film: the docstring of `docs/video/voices.py` records it as hexgrad's model under the Apache-2.0 licence, at https://huggingface.co/hexgrad/Kokoro-82M, 24 kHz, trained on permissively licensed and synthetic audio, and the film reads it through the `kokoro` and `huggingface_hub` packages.

**Data.** The **OpenADMET CYP blind challenge** is the source of the training data, the blinded test set and the definition of the TDI task; `README.md` records the space at https://huggingface.co/spaces/openadmet/cyp-challenge and the changelog of the data at https://huggingface.co/datasets/openadmet/cyp-challenge-train-test. The **Octant CYP inhibition blog release** is the quantitative panel that the structure track cross-checks against, recorded in `README.md` at https://huggingface.co/datasets/openadmet/Octant_CYP_inhibition_reactivity_blog_release and https://github.com/OpenADMET/Octant_CYP_blog_post under CC BY 4.0, and loaded by `cypmode/data/octant_benchmark.py`. The **Therapeutics Data Commons** is the source of the five-isoform CYP panel, recorded in `README.md` at https://tdc.hub.biolib.com and loaded by `cypmode/data/panel.py`; its NCATS qHTS measurements are the panel of Veith and co-workers (2009), and its Tox21 readings are the counter-screen of 9.10. The auxiliary measurements are the **ChEMBL**, **BindingDB** and **PubChem** readings loaded by `cypmode/challenge/chembl_auxiliary.py`, `bindingdb_auxiliary.py` and `pubchem_auxiliary.py`, the NCATS continuous measurements of `cypmode/challenge/external_continuous.py`, and the PXR-induction readings of the OpenADMET PXR challenge, published as `openadmet/pxr-challenge-train-test` under CC BY 4.0. The hydration free energies quoted in 10.5 are from the **FreeSolv** database of Mobley and Guthrie (2014). The reference complexes of the structure track are the **Protein Data Bank** entries that its benchmark lists.

The entries of this section are the record of what ran, not a second bibliography: the papers behind the models and the data are in C.1, and a version is the pin file's.
"""),
    code(LICENCE_TERMS),
    md("""**Terms behind the data record.** The files that the repository does not hold are packed into archives (12.1), and the policy of the record names, for each source and model behind them, what its licence says and where that was read. The table below is that policy: {{=n_terms}} entries, of which {{=n_terms_left_out}} are left out of the record, because the terms of what produced the files do not allow them to be redistributed under the licence of the record, which is CC BY 4.0, or because no terms are stated for them. One statement rests on the word of its authors and not on a licence file, and is named here so that nothing is left to be found: the parameter file that OpenFold3 uses by default is described as Apache 2.0 in the announcement of OpenBind-0, and no licence file for it was read.
"""),
    md("""*On what actually ran.* The works of C.1 were resolved against the records named in the introduction on 17 October 2026, one record per work, and the link at the end of an entry is that record; the few works whose record is a catalogue or an archive have no DOI, and the entry says where the record is. The table of C.2 reads `environment/` when the appendix is built, so it describes the environment beside it and not an older one, and the section names no program, model or data set that the notebook does not use. The appendix adds nothing to the text that the text does not cite or name.
"""),
]
