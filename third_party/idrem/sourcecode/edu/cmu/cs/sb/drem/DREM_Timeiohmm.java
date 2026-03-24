package edu.cmu.cs.sb.drem;

import edu.cmu.cs.sb.core.*;
import javax.swing.*;
import java.util.*;
import java.text.*;
import java.io.*;
import java.awt.*;
import java.math.*;
import java.awt.event.*;
import edu.umd.cs.piccolo.nodes.PText;

/**
 * This class implements the core methods for learning the DREM maps
 */
public class DREM_Timeiohmm {
	int nglobaltime;
	double nodepenalty = 10;
	boolean bmodeldiff = true;
	static final boolean BDEBUG = false;
	static final boolean BDEBUGMODEL = false;
	boolean ballowmergeval = false;
	boolean bsavedchange = false;
	boolean BEQUALSTD = false;
	boolean BVITERBI = false;
	boolean BREGDREM = false;
	boolean bhasmerge = false;
	double MINPROB = .00000000000000000001;
	double DEFAULTSIGMA = .5;
	double MAXFUTUREITR = 30;
	double MERGEMIN = -.0015;
	double DELAYMIN = -.0015;

	double MERGEMINDIFF = 0;
	double DELAYMINDIFF = 0;
	int nrandomseed;
	DREMGui progressDREMGui;
	int nglobaliteration;
	int ninitsearchval;

	double RESPLITMIN = -.0015;
	double RESPLITMINDIFF = 0;

	boolean brealXaxisDEF;
	double dYaxisDEF;
	double dXaxisDEF;
	double dnodekDEF;
	int nKeyInputTypeDEF;
	double dKeyInputXDEF;
	double dpercentDEF;
	JButton currentButton;

	JLabel statusLabel;
	JLabel statusLabel15;
	JLabel statuscountLabel;

	static final String SCORESDIR = "TOPAALoc";
	static final String SZDELIM = "|;,";
	DREM_NaiveBayes filteredClassifier;

	double dprevouterbestlog;
	/** last trainlikelihood */
	double dtrainlike;
	boolean bfilterbinding = false;
	double EPSILON = .01;
	double EPSILONDIFF = .01;
	double BEPSILON = 0.001;
	int nmaxchild;
	int ntotalcombined;// num filtered and non-filtered

	double[][] CONSTANTA;

	int MINPATH = 5;
	static final double RIDGE = 1;
	static final double MINSIGMA = .0001;
	double dminstddev = 0.5;

	String[] testgenenames;
	/**
	 * The subset of theDataSet.data that belongs to the training set??? The
	 * expression data: row are genes, columns are time points in experiments
	 */
	double[][] traindata;
	String[] traingenenames;
	double[][] testdata;

	int[][] trainpma;
	int[][] testpma;

	/**
	 * Object containing all of the Regulator-Gene binding data.
	 */
	RegulatorBindingData bindingData;
	/**
	 * Object containing the Regulator-Gene binding data for the train data set.
	 */
	RegulatorBindingData trainBinding;
	/**
	 * Object containing the Regulator-Gene binding data for the test data set.
	 */
	RegulatorBindingData testBinding;
	// /miRNA variables and tables
	// input variables from the IO
	// DREM_DataSet miRNADataSet;
	DataSetCore miRNADataSet;
        DataSetCore proteDataSet;
        
	String miRNAInteractionDataFile = "";
        String djPPIFile="";
	boolean scaleTFExp; // should TF expression be scaled?
	boolean scaleMIRNAExp; // should miRNA be used as a regulator
	HashMap<String, Integer> reg2DataSetIndex;
	HashMap<String, Integer> geneToDataSetIndex;

	double expressionScalingFactor = 1.0;
	double minTFExpAfterScaling = 0.5;
	boolean filterMIRNAExp = false;
	Treenode treeptr;
	NumberFormat nf2;
	NumberFormat nf3;
	double dbestlog;
	Treenode bestTree = null;
	DREM_DataSet theDataSet;

	JLabel statusLabel2;
	JLabel statusLabel3;
	int numtotalPath = 1;
	double dbesttrainlike;
        //proteomics and epigenetics weights
        String proteweight="1";
        
        

	/**
	 * Number of boolean items for the static vector. Equal to the number of TFs
	 */
	int numbits;
	double TRAINRATIO = .75;

	Random theRandom;

	int ntrain;
	int ntest;
	int numcols;
	int nholdout;

	double HOLDOUTRATIO = 0;
	double[][] holdoutdata;
	double[][][] holdoutpval;
	int[][] holdoutpma;
	int[][][] holdoutpvalIndex;

	boolean[] bholdout;
	BigInteger[] storedbestpath;
	String szinitfileval;
	String szexpname;
	String sznodepenaltyval;
	String szconvergenceval;
	String szbinding;
	ArrayList<Color> savedColors = new ArrayList<Color>();
	double dgloballike;
	private int ntotalID;

	Hashtable<Treenode, BackPtrRec> htBackPts = new Hashtable<Treenode, BackPtrRec>();

	String szstaticsourceval;

	String decodPath = "";
	String fastaDataFile = "";
	HashMap<String,String> gene2FastaMap;

	/**
	 * The probability that binding is functional, i.e. that the genes a TF
	 * binds are regulated by the TF. Used to calculated TF activity scores. Can
	 * be set in the DREM defaults file.
	 */
	double dProbBindingFunctional = 0.8;
	/**
	 * All TF activiy priors are adjusted by ACTIVITY_EPSILON to avoid divide by
	 * 0 errors when the TF activity prior is 1.
	 */
	static final double ACTIVITY_EPSILON = 1E-8;
	/**
	 * An array of activity scores for all TFs based on how the genes the TF
	 * regulates transition to the next states compared to how all genes into
	 * the split transistion. The max is the max score over this node and its
	 * descendent nodes.
	 */
	double[] dMaxTFActivityScore;
	String regPriorsFile = "";
        
  
        
        //get the metylation  score for each gene at each time point
        // Key: time+gene; Value: methy gene score;
        HashMap<String,HashMap<String,Double>> methyGeneScore;
	/**
	 * Class constructor - provides the execution control on the method
	 */
	public DREM_Timeiohmm(DREM_DataSet theDS, String szbinding,
			String sznumchildval, String szepsilonval, String szprunepathval,
			String szdelaypathval, String szmergepathval,
			String szepsilonvaldiff, String szprunepathvaldiff,
			String szdelaypathvaldiff, String szmergepathvaldiff,
			String szseedval, boolean bstaticcheckval, boolean ballowmergeval,
			int ninitsearchval, String szinitfileval, JLabel statusLabel,
			JLabel statusLabel15, JLabel statusLabel2, JLabel statusLabel3,
			JLabel statuscountLabel, JButton endSearchButton,
			boolean bstaticsearchval, final boolean brealXaxisDEF,
			final double dYaxisDEF, final double dXaxisDEF,
			final double dnodekDEF, final int nKeyInputTypeDEF,
			final double dKeyInputXDEF, final double dpercentDEF,
			final JButton currentButton, String szexpname,
			String sznodepenaltyval, boolean bpenalizedmodel,
			String szconvergenceval, String szminstddeval,
			String szstaticsourceval, boolean scaleTFExp,
			boolean scaleMIRNAExp, String miRNAInteractionDataFile,
                        String szExFile,
			DataSetCore miRNAExpressionDataCore, String miRNAExFile,  //mirna
                        DataSetCore proteDataCore, String djProteFile,String djPPIFile, Boolean bProteAll, //proteomics
                     
                        HashMap<String,HashMap<String,Double>> methyGeneScoreMap, // methy promoter score
                        String fastaDataFile,
			String decodPath, String regPriors, double dProbBindingFunctional,
			double expressionScalingFactor, double minTFExpAfterScaling,
			boolean filterMIRNAExp, String pweight )
			throws Exception {
		this.szstaticsourceval = szstaticsourceval;
		this.szbinding = szbinding;
		this.szconvergenceval = szconvergenceval;
		this.dminstddev = Double.parseDouble(szminstddeval);
		this.sznodepenaltyval = sznodepenaltyval;
		this.statusLabel = statusLabel;
		this.statusLabel15 = statusLabel15;
		this.statusLabel2 = statusLabel2;
		this.statusLabel3 = statusLabel3;
		this.statuscountLabel = statuscountLabel;
		this.szexpname = szexpname;
		this.ballowmergeval = ballowmergeval;
		this.brealXaxisDEF = brealXaxisDEF;
		this.dYaxisDEF = dYaxisDEF;
		this.dXaxisDEF = dXaxisDEF;
		this.dnodekDEF = dnodekDEF;
		this.nKeyInputTypeDEF = nKeyInputTypeDEF;
		this.dKeyInputXDEF = dKeyInputXDEF;
		this.dpercentDEF = dpercentDEF;
		this.currentButton = currentButton;
		this.ninitsearchval = ninitsearchval;
		this.scaleTFExp = scaleTFExp;
		this.scaleMIRNAExp = scaleMIRNAExp;
		this.miRNAInteractionDataFile = miRNAInteractionDataFile;
		this.decodPath = decodPath;
		this.dProbBindingFunctional = dProbBindingFunctional;
		this.expressionScalingFactor = expressionScalingFactor;
		this.minTFExpAfterScaling = minTFExpAfterScaling;
		this.filterMIRNAExp = filterMIRNAExp;
		this.fastaDataFile = fastaDataFile;
		this.regPriorsFile = regPriors;
                this.methyGeneScore=methyGeneScoreMap;
                this.proteweight=pweight;
               

		BREGDREM = (!bstaticsearchval);
               
		if (BDEBUG) {
			System.out.println(szepsilonval + "&&&&" + szseedval);
		}
                
		this.szinitfileval = szinitfileval;
		this.bfilterbinding = bstaticcheckval;

		nf2 = NumberFormat.getInstance(Locale.ENGLISH);
		nf2.setMinimumFractionDigits(2);
		nf2.setMaximumFractionDigits(2);

		nf3 = NumberFormat.getInstance(Locale.ENGLISH);
		nf3.setMinimumFractionDigits(3);
		nf3.setMaximumFractionDigits(3);

		this.statusLabel2 = statusLabel2;
		this.statusLabel3 = statusLabel3;

		// Why are these divided by 100???
		EPSILON = Double.parseDouble(szepsilonval) / 100;
		DELAYMIN = Double.parseDouble(szdelaypathval) / 100;
		RESPLITMIN = Double.parseDouble(szprunepathval) / 100;
		MERGEMIN = Double.parseDouble(szmergepathval) / 100;
		BEPSILON = Double.parseDouble(szconvergenceval) / 100;

		EPSILONDIFF = Double.parseDouble(szepsilonvaldiff);
		DELAYMINDIFF = Double.parseDouble(szdelaypathvaldiff);
		RESPLITMINDIFF = Double.parseDouble(szprunepathvaldiff);
		MERGEMINDIFF = Double.parseDouble(szmergepathvaldiff);

		nrandomseed = Integer.parseInt(szseedval);
		theRandom = new Random(nrandomseed);
		nodepenalty = Double.parseDouble(sznodepenaltyval);
		this.bmodeldiff = bpenalizedmodel;

		nmaxchild = Integer.parseInt(sznumchildval);

		theDataSet = theDS;
		HashMap<String,Integer> timepointName2Int = new HashMap<String,Integer>();
		for(int i = 0; i < theDataSet.dsamplemins.length; i++)
			timepointName2Int.put(theDataSet.dsamplemins[i].toUpperCase(), i);
		
                miRNADataSet = miRNAExpressionDataCore;
                proteDataSet = proteDataCore;
                
		CONSTANTA = new double[nmaxchild + 1][];
		for (int nindex = 1; nindex <= nmaxchild; nindex++) {
			CONSTANTA[nindex] = new double[nindex];
			for (int njindex = 0; njindex < nindex; njindex++)
				CONSTANTA[nindex][njindex] = 1.0 / nindex;
		}

		if (szbinding.equals("")) {
			BREGDREM = true;
		} else {
			/*
			 * When constructing RegulatorBindingData, if bfilterbinding is
			 * true, it is important to filter genes out of the data set.
			 */
			String mirnaFileToPass = "";
			if (bstaticsearchval)
				mirnaFileToPass = miRNAInteractionDataFile;
                        
			bindingData = new RegulatorBindingData(szbinding, mirnaFileToPass,
					bfilterbinding, theDataSet, SZDELIM, BDEBUG, regPriors,
					theDataSet.tga.szxrefval, timepointName2Int);
                        
			if (bfilterbinding) {
				theDataSet = new DREM_DataSet(theDataSet.filtergenesgeneral(
						bindingData.bbindingdata, bindingData.nbinding, true),
						theDataSet.tga);
			}
			
			// Create a hash map that takes regs and maps them to their row in
			// the exp data
			reg2DataSetIndex = new HashMap<String, Integer>();
			geneToDataSetIndex = new HashMap<String, Integer>();
			if (miRNADataSet != null) {
				for (int i = 0; i < miRNADataSet.genenames.length; i++)
					geneToDataSetIndex.put(miRNADataSet.genenames[i]
							.toUpperCase(), new Integer(i));
			}
			for (int i = 0; i < theDataSet.genenames.length; i++)
				geneToDataSetIndex.put(theDataSet.genenames[i].toUpperCase(),
						new Integer(i));
			for (int i = 0; i < bindingData.regNames.length; i++) {
				String upCaseReg = bindingData.regNames[i].toUpperCase();
				if (geneToDataSetIndex.containsKey(upCaseReg)) {
					reg2DataSetIndex.put(upCaseReg, geneToDataSetIndex
							.get(upCaseReg));
				} else {
					HashSet<String> syns = bindingData.regSyns.get(upCaseReg);
					boolean found = false;
					if (syns != null) {
						for (String syn : syns) {
							if (geneToDataSetIndex.containsKey(syn)) {
								if (found) {
									throw new Exception(
											"Multiple syns for "
													+ bindingData.regNames[i]
													+ ",a regulator in the"
													+ " binding data, are in the expression data.");
								}
								reg2DataSetIndex.put(upCaseReg,
										geneToDataSetIndex.get(syn));
								found = true;
							}
						}
					}
				}
			}
                        
                        //FIXME : remove not expressed miRNAs
                   
			if(filterMIRNAExp)
			{
				boolean[] keep = new boolean[bindingData.numberRegs];
				int count = 0;
				for(int i = 0; i < bindingData.numberRegs; i++)
				{
					if(bindingData.regTypes[i] == RegulatorBindingData.MIRNA)
					{
						if(reg2DataSetIndex.get(bindingData.regNames[i].toUpperCase()) == null)
						{
							keep[i] = false;
						}
						else //Has expression data
						{
							keep[i] = true;
							count++;
						}
					}
					else //TF
					{
						keep[i] = true;
						count++;
					}
				}
				bindingData = new RegulatorBindingData(bindingData,keep,count,false);
			}
			numbits = bindingData.numberRegs;

			// Parses the fasta formatted genes for use by DECOD
			if (fastaDataFile != null && !fastaDataFile.equals(""))
				parseFastaFile(fastaDataFile);

			//FIXME : adjustBIndingData
			bindingData
					.adjustBindingData(reg2DataSetIndex, 
                                                        theDataSet, szExFile,
							miRNADataSet,miRNAExFile,
                                                        proteDataSet,djProteFile,djPPIFile,bProteAll,
                                                        scaleMIRNAExp, scaleTFExp,
							expressionScalingFactor, minTFExpAfterScaling,
                                                        methyGeneScore,proteweight);
			buildFilteredClassifier();

			dMaxTFActivityScore = new double[numbits];
			for (int ntf = 0; ntf < bindingData.numberRegs; ntf++) {
				dMaxTFActivityScore[ntf] = 0;
			}
		}

		// compute mean and std at each time point of genes assigned to the
		// profile
		splitdata();
		treeptr = new Treenode();

		boolean binitfile = (!szinitfileval.equals(""));
		if ((binitfile) && (ninitsearchval == 0)) {
			readInitTree(treeptr);
			computeOrders(treeptr);
			combineTrainAndTest();
			viterbi(treeptr);
			if (bindingData.gene2RegBinding != null) {
				computeStats(treeptr, treeptr);
				computeminparentlevel(treeptr);
			}
			bsavedchange = true;
		} else {
			bsavedchange = false;
			if ((binitfile) && (ninitsearchval == 1)) {
				readInitTree(treeptr);
			} else {
				double[] dsigma = new double[traindata[0].length];
				double[] dmeans = new double[traindata[0].length];
				computeDataStats(traindata, trainpma, dsigma, dmeans);
				buildEmptyTree(0, treeptr, dsigma, dmeans);
			}

			searchstage1();
			int numintervals = traindata[0].length - 1;
			int[] path = new int[numintervals];

			int nremoved = 0;

			if (endSearchButton != null) {
				endSearchButton.setEnabled(false);
			}

			if (statusLabel3 != null) {
				statusLabel3.setText(" ");
			}

			if (statusLabel2 != null) {
				statusLabel2.setText(" ");
			}

			int NUMRESPLITS;
			if (bmodeldiff) {
				NUMRESPLITS = 0;
			} else {
				NUMRESPLITS = 1;
			}

			for (int ni = 1; ni <= NUMRESPLITS; ni++) {
				if (statusLabel3 != null) {
					statusLabel3.setText(" Deleting Paths: " + nremoved
							+ " removed so far ");
				}

				splitdata();
				if (BVITERBI) {
					dbestlog = trainhmmV(treeptr, true);
				} else {
					dbestlog = trainhmm(treeptr, true);
				}

				dbesttrainlike = dtrainlike;

				if (!bmodeldiff) {
					if (BVITERBI) {
						dbestlog = testhmmV(treeptr);
					} else {
						dbestlog = testhmm(treeptr);
					}
				}

				if (BDEBUG) {
					System.out.println("dbestlog = " + dbestlog);
				}

				boolean bdeleteagain = false;
				do {
					if (BDEBUG) {
						System.out.println("trying to delete after resplit");
					}

					traverse(bestTree, 0, false);
					dprevouterbestlog = dbestlog;
					dbestlog = Double.NEGATIVE_INFINITY;
					bdeleteagain = traverseanddelete(path, treeptr, treeptr,
							true, RESPLITMIN, RESPLITMINDIFF);
					if (dbestlog == Double.NEGATIVE_INFINITY)
						dbestlog = dprevouterbestlog;

					if (BDEBUG) {
						System.out
								.println("****** delete  best log likelihood "
										+ dbestlog);
					}

					traverse(bestTree, 0, true);
					if (bestTree != treeptr) {
						nremoved++;
						numtotalPath--;
						if (statuscountLabel != null) {
							statuscountLabel
									.setText(" Number of paths in model so far: "
											+ numtotalPath);
							statusLabel3.setText(" Deleting Paths: " + nremoved
									+ " removed so far");
							statusLabel2.setText(" ");
						}
						treeptr = bestTree;
					}

					if (BDEBUG) {
						System.out.println("CCC\tdelete\t" + dbesttrainlike
								+ "\t" + dbestlog + "\t" + dprevouterbestlog);
					}
				} while (bdeleteagain);
			}

			boolean bagaindelay;
			boolean bagaindelayouter;

			int numdelay = 0;

			do {
				bagaindelayouter = false;
				for (int ndesiredlevel = 1; ndesiredlevel < numintervals; ndesiredlevel++) {
					do {
						if (!bmodeldiff) {
							if (BVITERBI) {
								dbestlog = testhmmV(treeptr);
							} else {
								dbestlog = testhmm(treeptr);
							}
						}

						if (BDEBUG) {
							System.out.println("trying to delay "
									+ ndesiredlevel + " " + dbestlog);
						}

						dprevouterbestlog = dbestlog;

						dbestlog = Double.NEGATIVE_INFINITY;
						bagaindelay = traverseanddelay(path, treeptr,
								ndesiredlevel, treeptr, false);

						if (dbestlog == Double.NEGATIVE_INFINITY) {
							dbestlog = dprevouterbestlog;
						} else {
							numdelay++;
							if (statusLabel3 != null) {
								statusLabel3.setText(" Number of delayed is "
										+ numdelay + " so far");
							}
							treeptr = bestTree;

							double dimprovedelay = (dbestlog - dprevouterbestlog)
									/ Math.abs(dbestlog);
							String szimprovedelay = nf3
									.format(100 * dimprovedelay)
									+ "%";
							String szimprovedelayDIFF = nf3.format(dbestlog
									- dprevouterbestlog);

							if (statusLabel2 != null) {
								statusLabel2.setText(" Delay improvement: "
										+ szimprovedelayDIFF + " ("
										+ szimprovedelay + ")");
							}
						}

						if (BDEBUG) {
							System.out
									.println("****** delay  best log likelihood "
											+ dbestlog);
						}
						traverse(bestTree, 0, false);

						treeptr = bestTree;

						if (BDEBUG) {
							System.out.println("YYY\tdelay\t" + dbesttrainlike
									+ "\t" + dbestlog + "\t"
									+ dprevouterbestlog);
						}

						if (bagaindelay) {
							bagaindelayouter = true;
						}
					} while (bagaindelay);
				}
			} while (bagaindelayouter);

			if (ballowmergeval) {
				traverseandmerge(path);
			}
			treeptr = bestTree;

			traverse(bestTree, 0, false);
			if (BDEBUG) {
				System.out.println("Test set " + dbestlog);
				System.out.println("calling combine tranandtest");
			}

			combineTrainAndTest();

			if (BVITERBI) {
				trainhmmV(treeptr, true);
			} else {
				double dtemplog = trainhmm(treeptr, true);
				nglobaltime++;
				int numnodes = countNodes(treeptr, nglobaltime);
				if (BDEBUGMODEL) {
					System.out.println(nodepenalty + "\t final train is\t"
							+ dtemplog + "\t" + numnodes + "\t"
							+ (dtemplog + numnodes * nodepenalty) + "\t"
							+ dgloballike);
				}
			}
			traverse(bestTree, 0, false);

			boolean bagain;

			do {
				bagain = false;

				computeOrders(treeptr);
				viterbi(treeptr);

				int[] bestpath = new int[path.length];
				for (int nindex = 0; nindex < bestpath.length; nindex++) {
					bestpath[nindex] = -1;
				}

				MinPathRec theMinPathRec = traverseanddeleteMinPath(path,
						bestpath, treeptr);
				if (BDEBUG) {
					System.out.println("min is " + theMinPathRec.nval + "\t"
							+ "level is " + theMinPathRec.nlevel);
					for (int nindex = 0; nindex < bestpath.length; nindex++) {
						System.out.println(theMinPathRec.bestpath[nindex]);
					}
				}

				if (theMinPathRec.nval < MINPATH) {
					deleteMinPath(theMinPathRec.bestpath, theMinPathRec.nlevel,
							treeptr);
					bagain = true;
					traverse(treeptr, 0, true);

					if (BVITERBI) {
						trainhmmV(treeptr, true);
					} else {
						trainhmm(treeptr, true);
					}

					if (BDEBUG) {
						System.out.println("after retrain");
					}

					traverse(treeptr, 0, true);

					nremoved++;
					numtotalPath--;
					if (statuscountLabel != null) {
						statuscountLabel
								.setText(" Number of paths in model so far: "
										+ numtotalPath);
						statusLabel3.setText(" Deleting Paths: " + nremoved
								+ " removed so far");
						statusLabel2.setText(" ");
					}
				}
			} while (bagain);
			if (bindingData.gene2RegBinding != null) {
				computeStats(treeptr, treeptr);
				computeminparentlevel(treeptr);
			}
		}
	}

	/**
	 * Sets up the stored data so that the training data includes all data
	 * except any external held out validation data, meaning the data which
	 * might otherwise used for model selection is just for parameter learning
	 * 
	 * @throws Exception
	 */
	public void combineTrainAndTest() throws Exception {
		nholdout = (int) (theDataSet.data.length * HOLDOUTRATIO);
		ntrain = (theDataSet.data.length - nholdout);
		numcols = theDataSet.numcols;

		traindata = new double[ntrain][numcols];

		trainpma = new int[ntrain][numcols];
		traingenenames = new String[ntrain];

		int ntrainindex = 0;
		boolean[] toKeep = new boolean[theDataSet.data.length];

		for (int nindex = 0; nindex < theDataSet.data.length; nindex++) {
			if (!bholdout[nindex]) {
				for (int ncol = 0; ncol < numcols; ncol++) {
					traindata[ntrainindex][ncol] = theDataSet.data[nindex][ncol];
					trainpma[ntrainindex][ncol] = theDataSet.pmavalues[nindex][ncol];
				}
				traingenenames[ntrainindex] = theDataSet.genenames[nindex];
				toKeep[nindex] = true;
				ntrainindex++;
			}
		}
		trainBinding = new RegulatorBindingData(bindingData, toKeep, ntrain, true);
		trainBinding.loadInstances(nmaxchild, numbits);
	}

	/**
	 * Splits the non-held out validation data into a training set for learning
	 * the model parameters and a test set for model selection
	 * 
	 * @throws Exception
	 */
	public void splitdata() throws Exception {
		// splits data into training and testsets

		nholdout = (int) (theDataSet.data.length * HOLDOUTRATIO);
		if (bmodeldiff) {
			ntrain = (theDataSet.data.length - nholdout);
			ntest = 0;
		} else {
			ntrain = (int) ((theDataSet.data.length - nholdout) * TRAINRATIO);
			ntest = theDataSet.data.length - ntrain - nholdout;
		}

		numcols = theDataSet.data[0].length;

		// build arrays with time-series training data
		// and binding p-value
		testgenenames = new String[ntest];

		testdata = new double[ntest][numcols];
		testpma = new int[ntest][numcols];
		traindata = new double[ntrain][numcols];
		trainpma = new int[ntrain][numcols];

		double[] foldrandom = new double[theDataSet.data.length];
		double[] foldrandomcopy = new double[theDataSet.data.length];

		// If a set of genes to be held out does not exist yet, create one
		if (bholdout == null) {
			holdoutdata = new double[nholdout][numcols];
			bholdout = new boolean[theDataSet.data.length];

			holdoutpval = new double[numcols][nholdout][numbits];

			holdoutpma = new int[nholdout][numcols];
			holdoutpvalIndex = new int[numcols][nholdout][];

			// Assign random values and store a copy of them
			for (int nindex = 0; nindex < foldrandom.length; nindex++) {
				foldrandom[nindex] = theRandom.nextDouble();
				foldrandomcopy[nindex] = foldrandom[nindex];
			}
			Arrays.sort(foldrandomcopy);
			double dcutoff = foldrandomcopy[holdoutpval[0].length];

			int nholdoutindex = 0;

			for (int nindex = 0; nindex < foldrandom.length; nindex++) {
				// The cutoff was chosen to ensure that the desired
				// number of genes are held out
				if (foldrandom[nindex] < dcutoff) {
					for (int ncol = 0; ncol < numcols; ncol++) {
						holdoutpval[ncol][nholdoutindex] = bindingData.gene2RegBinding[ncol][nindex];
						holdoutpvalIndex[ncol][nholdoutindex] = bindingData.gene2RegBindingIndex[ncol][nindex];
						holdoutdata[nholdoutindex][ncol] = theDataSet.data[nindex][ncol];
						holdoutpma[nholdoutindex][ncol] = theDataSet.pmavalues[nindex][ncol];
					}
					bholdout[nindex] = true;
					nholdoutindex++;
				} else {
					bholdout[nindex] = false;
				}
			}
		}

		/*---------------------------------------------*/
		// random drawing a set of ntrain elements from data.length elements
		/*
		 * The method for drawing the random elements is to maintain a set of
		 * ntrain elements marked as true while iterating through the elements.
		 * The set starts with the initial ntrain elements and adds each element
		 * afterwards with probability (ntrain/nonholdout) swapping a random
		 * element out when this happens.
		 */
		int[] includetrain = new int[ntrain];
		boolean[] btrain = new boolean[theDataSet.data.length];
		if (BDEBUG) {
			System.out.println(ntrain + " " + theDataSet.data.length);
		}

		int nincludeindex = 0;
		int nbtrainindex = 0;
		int nonholdout = 0;
		while (nincludeindex < ntrain) {
			if (!bholdout[nbtrainindex]) {
				includetrain[nincludeindex] = nbtrainindex;
				btrain[nbtrainindex] = true;
				nincludeindex++;
				nonholdout++;
			}
			nbtrainindex++;
		}
		if (BDEBUG) {
			System.out.println("done");
		}

		// drawing nrandom elements from a set of numtotalannotated elements
		// where each element is equally likely
		for (; nbtrainindex < btrain.length; nbtrainindex++) {
			if (!bholdout[nbtrainindex]) {
				if (theRandom.nextDouble() < ((double) ntrain / (double) (nonholdout + 1))) {
					int nreplaceindex = (int) Math.floor(ntrain
							* theRandom.nextDouble());
					btrain[nbtrainindex] = true;
					btrain[includetrain[nreplaceindex]] = false;
					includetrain[nreplaceindex] = nbtrainindex;
				} else {
					btrain[nbtrainindex] = false;
				}
				nonholdout++;
			}
		}
		/*------------------------------------------------------*/
		trainBinding = new RegulatorBindingData(bindingData, btrain, ntrain, true);
		trainBinding.loadInstances(nmaxchild, numbits);

		int ntrainindex = 0;
		int ntestindex = 0;
		for (int nindex = 0; nindex < btrain.length; nindex++) {
			if (!bholdout[nindex]) {
				if (btrain[nindex]) {
					for (int ncol = 0; ncol < numcols; ncol++) {
						traindata[ntrainindex][ncol] = theDataSet.data[nindex][ncol];
						trainpma[ntrainindex][ncol] = theDataSet.pmavalues[nindex][ncol];
					}
					ntrainindex++;

				} else {
					testgenenames[ntestindex] = theDataSet.genenames[nindex];

					for (int ncol = 0; ncol < numcols; ncol++) {
						testdata[ntestindex][ncol] = theDataSet.data[nindex][ncol];
						testpma[ntestindex][ncol] = theDataSet.pmavalues[nindex][ncol];
					}
					ntestindex++;
				}
			}
		}
		// Flip the boolean keep array to create test data
		for (int i = 0; i < btrain.length; i++) {
			if (!bholdout[i])
				btrain[i] = !btrain[i];
		}
		testBinding = new RegulatorBindingData(bindingData, btrain,
				theDataSet.data.length - ntrain - nholdout, true);
	}

	/**
	 * Parses a file specified from the GUI to obtain the FASTA format of genes
	 * for use by DECOD.
	 * 
	 * @param file
	 * @throws IOException
	 */
	public void parseFastaFile(String file) throws IOException {
		gene2FastaMap = new HashMap<String, String>();
		BufferedReader br = new BufferedReader(new FileReader(file));
		String gene;
		String fastaFormat = "";
		
		String line = br.readLine();
		while(line != null && line.charAt(0)=='#')
			line = br.readLine();
		
		while (line != null) {
			if (line.charAt(0) != '>')
				throw new IOException();
			gene = line.substring(1);
			fastaFormat = line + "\n";
			line = br.readLine();
			while (line != null && line.charAt(0) != '>') {
				fastaFormat += line + "\n";
				
				line = br.readLine();
				while(line != null && line.charAt(0)=='#')
					line = br.readLine();
			}
			gene2FastaMap.put(gene.toUpperCase(), fastaFormat);
		}
	}

	/**
	 * Computes the average and standard deviation expression level at each time
	 * point
	 */
	public void computeDataStats(double[][] data, int[][] pma,
			double[] dsigmas, double[] dmeans) {
		double dsum;
		double dsumsq;
		int npoints;
		double dsigmaavg = 0;
		for (int ncol = 0; ncol < data[0].length; ncol++) {
			dsum = 0;
			dsumsq = 0;
			npoints = 0;
			// need to handle 0 or 1 point error
			for (int nrow = 0; nrow < data.length; nrow++) {
				if (pma[nrow][ncol] != 0) {
					dsum += data[nrow][ncol];
					dsumsq += Math.pow(data[nrow][ncol], 2);
					npoints++;
				}
			}
			dmeans[ncol] = dsum / npoints;
			dsigmas[ncol] = Math.sqrt((dsumsq - Math.pow(dsum, 2) / npoints)
					/ (npoints - 1));
			if (BDEBUG) {
				System.out.print(dsigmas[ncol] + "\t");
			}
			dsigmaavg += dsigmas[ncol];
			if (BDEBUG) {
				System.out.println("###" + dsigmas[ncol]);
			}
		}
		DEFAULTSIGMA = dsigmaavg / data[0].length;
		if (BDEBUG) {
			System.out.println("[[[]]]" + DEFAULTSIGMA);
			System.out.println();
		}
	}

	// /////////////////////////////////////////////////
	/**
	 * Uses DREM_NaiveBayes to build a classifier which predicts given the set
	 * of transcription factors predicted to regulate a gene whether it would
	 * appear filtered or not
	 */
	public void buildFilteredClassifier() {
		Iterator<String> itrgenes = theDataSet.htFiltered.keySet().iterator();
		int numfiltered = theDataSet.htFiltered.size();

		int[][] filteredinput = new int[numfiltered][];
		int[][] filteredinputIndex = new int[numfiltered][];

		int nfilteredhits = 0;
		int[] ALLZEROES = new int[0];

		int nindex = 0;
		while (itrgenes.hasNext()) {
			String szgene = itrgenes.next();
			String szprobe = (String)theDataSet.htFiltered.get(szgene);

			Integer rbdindex = geneToDataSetIndex.get(szgene);
			if (rbdindex == null)
				rbdindex = geneToDataSetIndex.get(szprobe);

			if (rbdindex != null) {
				int[] filteredGeneRecInput = new int[bindingData.gene2RegMaxBinding[rbdindex].length];
				for (int i = 0; i < bindingData.gene2RegMaxBinding[rbdindex].length; i++) {
					if (bindingData.gene2RegMaxBinding[rbdindex][i] > 0)
						filteredGeneRecInput[i] = 1;
					else if (bindingData.gene2RegMaxBinding[rbdindex][i] < 0)
						filteredGeneRecInput[i] = -1;
					else
						filteredGeneRecInput[i] = 0;
				}
				filteredinput[nindex] = filteredGeneRecInput;
				filteredinputIndex[nindex] = bindingData.gene2RegMaxBindingIndex[rbdindex];
				nfilteredhits++;
			} else {
				if (!bfilterbinding) {
					// not filtering binding using all zeros insted
					filteredinput[nindex] = ALLZEROES;
					filteredinputIndex[nindex] = ALLZEROES; // really empty
				} else {
					filteredinput[nindex] = null;
					filteredinputIndex[nindex] = null;
				}
			}
			nindex++;
		}

		int nfinalfilter;
		if (bfilterbinding) {
			nfinalfilter = nfilteredhits;
		} else {
			nfinalfilter = numfiltered;
		}

		ntotalcombined = bindingData.gene2RegMaxBinding.length + nfinalfilter;

		if (BDEBUG) {
			System.out.println(bindingData.gene2RegMaxBinding.length
					+ " $$$$$$$$$$ " + nfinalfilter);
		}
		// making new array with filtered and non-filtered
		// need to do this TF wise
		int[][] combinedbinding = new int[ntotalcombined][];
		int[][] combinedbindingIndex = new int[ntotalcombined][];
		int[] filteredlabel = new int[ntotalcombined];
		double[] trainweight = new double[ntotalcombined];
		for (int ni = 0; ni < bindingData.gene2RegMaxBinding.length; ni++) {
			filteredlabel[ni] = 1;
			combinedbinding[ni] = new int[bindingData.gene2RegMaxBinding[ni].length];
			for (int i = 0; i < combinedbinding[ni].length; i++) {
				combinedbinding[ni][i] = (int) Math
						.signum(bindingData.gene2RegMaxBinding[ni][i]);
			}
			combinedbindingIndex[ni] = bindingData.gene2RegMaxBindingIndex[ni];
		}

		for (int ni = bindingData.gene2RegMaxBinding.length; ni < filteredlabel.length; ni++) {
			filteredlabel[ni] = 0;
		}

		int nfilteredindex = 0;
		int ntotalindex = bindingData.gene2RegMaxBinding.length;
		while ((nfilteredindex < filteredinput.length)
				&& (ntotalindex < combinedbinding.length)) {
			if (filteredinput[nfilteredindex] != null) {
				combinedbinding[ntotalindex] = filteredinput[nfilteredindex];
				combinedbindingIndex[ntotalindex] = filteredinputIndex[nfilteredindex];
				ntotalindex++;
			}
			nfilteredindex++;
		}

		for (int ni = 0; ni < trainweight.length; ni++) {
			trainweight[ni] = 1;
		}

		// combinedbinding input values
		// filtered label y-labels
		if (bindingData.signedBindingValuesSorted.length > 0) {
			filteredClassifier = new DREM_NaiveBayes(combinedbinding,
					combinedbindingIndex, numbits, filteredlabel,
					bindingData.signedBindingValuesSorted, 2);
		} else {
			filteredClassifier = null;
		}

		if (BDEBUG) {
			System.out.println(filteredClassifier);
		}
	}

	// //////////////////////////////////////////////////
	/**
	 * Displays the current temporary DREM model, from which a search is
	 * performed for an improved model
	 */
	public void displayTempMap() throws Exception {
		DREM_IO.bdisplaycurrent = false;
		final DREM_Timeiohmm fthishmm = this;
		if (BDEBUG) {
			System.out.println("before hmmgui");
		}

		final Treenode treecopy = (Treenode) treeptr.clone();

		computeOrders(treecopy);

		viterbi(treecopy);
		if (bindingData.gene2RegBinding != null) {
			computeStats(treecopy, treecopy);
			computeminparentlevel(treecopy);
		}

		((DREM_GoAnnotations) theDataSet.tga).buildRecDREM(treecopy,
				theDataSet.genenames);

		traverse(bestTree, 0, false);

		try {
			javax.swing.SwingUtilities.invokeAndWait(new Runnable() {
				public void run() {
					progressDREMGui = new DREMGui(fthishmm, treecopy,
							brealXaxisDEF, dYaxisDEF, dXaxisDEF,
							nKeyInputTypeDEF, dKeyInputXDEF, dpercentDEF,
							"(Not Final Model)", dnodekDEF);
					edu.umd.cs.piccolo.PCanvas.CURRENT_ZCANVAS = null;
					progressDREMGui.setLocation(15, 40);
					progressDREMGui
							.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

					progressDREMGui.setVisible(true);
					progressDREMGui.addWindowListener(new WindowAdapter() {
						public void windowClosing(WindowEvent we) {
							progressDREMGui.closeWindows();
						}
					});
				}
			});
		} catch (InterruptedException iex) {
			iex.printStackTrace(System.out);
		} catch (java.lang.reflect.InvocationTargetException itex) {
			itex.printStackTrace(System.out);
		}
		if (currentButton != null) {
			currentButton.setEnabled(true);
		}

	}

	// ////////////////////////////////////////////////
	/**
	 * Executes the first phase of the structure search Considers adding and
	 * deleting paths, does not consider merges or delays
	 */
	public void searchstage1() throws Exception {
		String szDIFF = "";
		String szpercent = "";
		int numintervals = traindata[0].length - 1;
		int[] path = new int[numintervals];

		traverse(treeptr, 0, false);

		if (BVITERBI) {
			dbestlog = trainhmmV(treeptr, true);
		} else {
			dbestlog = trainhmm(treeptr, true);
		}
		traverse(treeptr, 0, false);

		// ////////////////////////////

		double dprevbestlog;
		dprevouterbestlog = Double.NEGATIVE_INFINITY;

		if (!bmodeldiff) {
			if (BVITERBI) {
				dbestlog = testhmmV(treeptr);
			} else {
				dbestlog = testhmm(treeptr);
			}
		}
		dbesttrainlike = Double.NEGATIVE_INFINITY;
		boolean bendsearchlocal;

		do {
			if (BDEBUG) {
				System.out.println("&&&& " + numtotalPath);
			}
			// outer loop handles the re-splitting of the data
			// and whether we should try to add and then delete nodes

			if (dprevouterbestlog != Double.NEGATIVE_INFINITY) {

				if (statusLabel != null) {
					statusLabel.setText(" Current score: "
							+ nf2.format(dbestlog));
					statusLabel15.setText(" Current score improvement: "
							+ szDIFF + " (" + szpercent + ")");
					statusLabel3
							.setText(" Next score: " + nf2.format(dbestlog));
					statusLabel2.setText(" Next score improvement: 0 (0.000%)");
					statuscountLabel
							.setText(" Number of paths in model so far: "
									+ numtotalPath);
				}
			}
			dprevouterbestlog = dbestlog;

			traverse(treeptr, 0, false);

			bestTree = treeptr;

			if (BDEBUG) {
				System.out.println("trying to add");
			}
			dprevbestlog = dbestlog;

			traverse(bestTree, 0, false);
			traverseandadd(path, treeptr, treeptr);
			if (BDEBUG) {
				System.out.println("****** adding best log likelihood "
						+ dbestlog);
			}
			traverse(bestTree, 0, false);

			if (BDEBUG) {
				System.out.println("after traverse");
			}
			treeptr = bestTree;

			if (BDEBUG) {
				System.out.println("$$$$$$$$$$$$$$$terminate " + dbestlog
						+ "\t" + dprevbestlog + "\t"
						+ (dbestlog - dprevbestlog) / Math.abs(dbestlog) + "\t"
						+ (dbestlog - dprevbestlog));
			}

			if (BDEBUG) {
				System.out.println("YYY\tadd\t" + dbesttrainlike + "\t"
						+ dbestlog);
			}

			if (dbestlog > dprevbestlog) {
				numtotalPath++;
			}

			while ((dbestlog - dprevbestlog) > 0) {
				if (BDEBUG) {
					System.out.println("trying to delete");
				}

				dprevbestlog = dbestlog;

				traverseanddelete(path, treeptr, treeptr, false, 0, 0);

				if (BDEBUG) {
					System.out.println("****** delete  best log likelihood "
							+ dbestlog);
				}

				traverse(bestTree, 0, false);

				treeptr = bestTree;

				if (BDEBUG) {
					System.out.println("YYY\tdelete\t" + dbesttrainlike + "\t"
							+ dbestlog);
				}

				if (dbestlog > dprevbestlog) {
					numtotalPath--;
				}
			}

			if (BDEBUG) {
				System.out.println("after delete " + dbestlog + "\t"
						+ dprevouterbestlog);
				System.out.println("*************terminate "
						+ (dbestlog - dprevouterbestlog) + "\t"
						+ (dbestlog - dprevouterbestlog));
				System.out.println("ZZZ\t" + dbesttrainlike + "\t" + dbestlog);
			}

			szpercent = nf3.format(100 * (dbestlog - dprevouterbestlog)
					/ Math.abs(dbestlog))
					+ "%";
			szDIFF = nf3.format(dbestlog - dprevouterbestlog);

			bendsearchlocal = DREM_IO.bendsearch;

			if (DREM_IO.bdisplaycurrent) {
				displayTempMap();
			}

			if (BDEBUG) {
				System.out.println("$" + EPSILON + " " + dbestlog + " "
						+ dprevouterbestlog + " " + EPSILONDIFF);
			}

		} while ( ((dbestlog - dprevouterbestlog)/ Math.abs(dbestlog)>BEPSILON) && ((((dbestlog - EPSILON * Math.abs(dbestlog) - dprevouterbestlog) > EPSILONDIFF)
				&& (!bendsearchlocal) && !bmodeldiff)
				|| (bmodeldiff && (dbestlog > dprevouterbestlog) && (!bendsearchlocal))));

		if (statusLabel15 != null) {
			statusLabel15.setText(" Final Main Score Improvement: " + szDIFF
					+ " (" + szpercent + ")");
			statusLabel.setText(" Final Main Score: " + nf2.format(dbestlog));
			statuscountLabel.setText(" Number of paths in model so far: "
					+ numtotalPath);
		}
	}

	// //////////////////////////////////////////////////////////////////////

	/**
	 * Assigns to the nminparentlevel in all nodes the level of the most
	 * immediate ancestor with two or more children
	 */
	public void computeminparentlevel(Treenode ptr) {
		if (ptr != null) {
			if ((ptr.parentptrA == null) || (ptr.parentptrA.length == 1)) {
				if (ptr.parent == null) {
					ptr.nminparentlevel = ptr.ndepth;
					ptr.minparentptr = ptr.parent;
				} else {
					if (ptr.parent.numchildren >= 2) {
						ptr.nminparentlevel = ptr.ndepth;
						ptr.minparentptr = ptr.parent;
					} else {
						ptr.nminparentlevel = ptr.parent.nminparentlevel;
						ptr.minparentptr = ptr.parent.minparentptr;
					}
				}
			} else {
				Treenode[] tempptrA = new Treenode[ptr.parentptrA.length];
				int ndepth = ptr.ndepth;
				for (int nindex = 0; nindex < tempptrA.length; nindex++) {
					tempptrA[nindex] = ptr.parentptrA[nindex];
				}

				boolean bfoundit = false;
				while (!bfoundit) {
					int nj = 1;
					boolean ballsame = true;
					while ((ballsame) && nj < tempptrA.length) {
						if (tempptrA[nj] != tempptrA[nj - 1]) {
							ballsame = false;
						} else {
							nj++;
						}
					}

					ndepth--;
					bfoundit = ballsame;
					for (int nindex = 0; nindex < tempptrA.length; nindex++) {
						tempptrA[nindex] = tempptrA[nindex].parent;
					}
				}

				ptr.nminparentlevel = ndepth;
				ptr.minparentptr = tempptrA[0];
			}

			for (int nindex = 0; nindex < ptr.nextptr.length; nindex++) {
				computeminparentlevel(ptr.nextptr[nindex]);
			}
		}
	}

	/**
	 * Responsible with helper functions for merging paths. Paths must share a
	 * common most recent split and no splits after a merge are allowed
	 */
	public void traverseandmerge(int[] path) throws Exception {

		computeNumLeaves(bestTree);
		if (BDEBUG) {
			System.out.println("in traverse and merge path.length = "
					+ path.length);
		}

		traverse(bestTree, 0, false);

		for (int nparentlevel = path.length - 2; nparentlevel >= 0; nparentlevel--) {
			for (int nmergingdepth = path.length; nmergingdepth > nparentlevel + 1; nmergingdepth--) {
				// nmergingdepth is the depth of the merging
				// parent level is the common ancestor
				Treenode origbestTree;
				do {
					dprevouterbestlog = dbestlog;
					dbestlog = Double.NEGATIVE_INFINITY;

					if (BDEBUG) {
						System.out.println("calling traverseandmergehelp "
								+ nmergingdepth + " " + nparentlevel);
					}
					origbestTree = bestTree;
					traverseandmergehelp(path, 0, nmergingdepth, nparentlevel,
							origbestTree, origbestTree);
					if (dbestlog == Double.NEGATIVE_INFINITY)
						dbestlog = dprevouterbestlog;
				} while (origbestTree != bestTree);
			}
		}

	}

	/**
	 * Helper function for traverseandmerge used to recursively search for
	 * places to merge nodes
	 */
	private void traverseandmergehelp(int[] path, int nlevel,
			int nmergingdepth, int nparentlevel, Treenode ptr, Treenode origroot)
			throws Exception {

		if (nlevel <= nmergingdepth) {
			for (int nindex = 0; nindex < ptr.numchildren; nindex++) {
				path[nlevel] = nindex;
				traverseandmergehelp(path, nlevel + 1, nmergingdepth,
						nparentlevel, ptr.nextptr[nindex], origroot);
			}
		}

		if (nlevel == nmergingdepth) {
			if (BDEBUG) {
				System.out.println("nparentlevel = " + nparentlevel
						+ " nmergindepth = " + nmergingdepth + " nlevel = "
						+ nlevel + " mean = " + ptr.dmean);
			}
			mergenode(path, nmergingdepth, nparentlevel, origroot);
		}
	}

	/**
	 * Computes the number of paths to leaves accessible from each node in the
	 * tree pointed to by ptr
	 */
	public int computeNumLeaves(Treenode ptr) {
		if (ptr.numchildren == 0) {
			return 1;
		} else {
			int nsum = 0;
			for (int nindex = 0; nindex < ptr.numchildren; nindex++) {
				nsum += computeNumLeaves(ptr.nextptr[nindex]);
			}

			if (BDEBUG) {
				System.out.println("ptr.dmean = " + ptr.dmean + " nsum = "
						+ nsum);
			}

			ptr.numdownleaves = nsum;

			return nsum;
		}
	}

	// ////////////////////////////////////////////////////////////////////////////////////////
	/**
	 * Returns true iff all nodes at nleveltogo only have one path to the leaves
	 * Stores in hsLevel all nodes at nleveltogo is 0
	 */
	public boolean findnodesAtLevel(Treenode treeptr, int nleveltogo,
			HashSet<Treenode> hsLevel) {
		if (nleveltogo == 0) {
			hsLevel.add(treeptr);
			if (treeptr.numdownleaves >= 2)
				return false;
			else
				return true;
		} else {
			boolean ballsingles = true;
			for (int nindex = 0; nindex < treeptr.numchildren; nindex++) {
				ballsingles = (ballsingles)
						&& (findnodesAtLevel(treeptr.nextptr[nindex],
								nleveltogo - 1, hsLevel));
			}
			return ballsingles;
		}
	}

	// ////////////////////////////////////////////////////////////////////////////////////////////////
	/**
	 * Helper function of traverseandmergehelp responsible for finding valid
	 * merges and evaluating them that are along path1 and at depth
	 * nmergingdepth
	 */
	private void mergenode(int[] path1, int nmergingdepth, int nparentlevel,
			Treenode origroot) throws Exception {
		double dlog;

		if (BDEBUG) {
			System.out.println("nmerginglevel = " + nmergingdepth);
			System.out.println("nparentlevel = " + nparentlevel);
			System.out.print("Path: ");
			for (int i = 0; i < nparentlevel; i++) {
				System.out.print(path1[i] + " ");
			}
			System.out.println();
		}

		Treenode tempptr = origroot;
		for (int nindex = 0; nindex < nparentlevel; nindex++) {
			tempptr = tempptr.nextptr[path1[nindex]];
		}

		if ((tempptr.numchildren >= 2)) {
			HashSet<Treenode> hsLevel = new HashSet<Treenode>();
			boolean ballsingles = findnodesAtLevel(tempptr,
					(nmergingdepth - nparentlevel), hsLevel);
			// ballsingles - is true if no more splits among nodes being merged

			int numatLevel = hsLevel.size();
			// only binary for now
			if (BDEBUG) {
				System.out.println("numatLevel = " + numatLevel);
			}

			if ((numatLevel == tempptr.numchildren) && (ballsingles)) {
				// not allowing any downstream splits from parent node
				// either before the level being merged, should have been
				// accepted before
				// among merged nodes, not allowing any resplits
				// require ballsingles - to be true
				// also cannot have any splits
				// assume binary for now

				if (BDEBUG) {
					System.out.println("before clone");
				}

				traverse(origroot, 0, false);
				Treenode treeroot = (Treenode) origroot.clone();
				if (BDEBUG) {
					System.out.println("after clone");
				}
				traverse(treeroot, 0, false);

				// traversing to the parent
				tempptr = treeroot;
				for (int nindex = 0; nindex < nparentlevel; nindex++) {
					tempptr = tempptr.nextptr[path1[nindex]];
				}

				Treenode parentptr = tempptr;

				// travering to the nodes for merging
				Treenode splitptr = null;
				for (int nindex = nparentlevel; nindex < nmergingdepth; nindex++) {
					tempptr = tempptr.nextptr[path1[nindex]];
				}

				if (BDEBUG) {
					System.out.println(nmergingdepth + " parent mean "
							+ parentptr.dmean + " tempptr.dmean "
							+ tempptr.dmean + " split mean ");
				}

				boolean bvalidmerge = false;
				ArrayList<Treenode> newparentptrA = null;
				for (int notherchild = 0; notherchild < parentptr.numchildren; notherchild++) {
					if (BDEBUG) {
						System.out.println(notherchild + "\t" + bvalidmerge);
					}

					if (notherchild != path1[nparentlevel]) {
						splitptr = parentptr.nextptr[notherchild];

						for (int nindex = nparentlevel + 1; nindex < nmergingdepth; nindex++) {
							splitptr = splitptr.nextptr[0];
						}

						if (splitptr != tempptr) {
							// its possible two nodes have already been merged
							// on this level
							// but there are two paths two them, we do not need
							// to adjust parent
							// pointers if just taking a different path to the
							// main merging node
							if (!bvalidmerge) {
								// we haven't tried merging yet, just do this
								// once to load the merging node
								// parents into the list
								newparentptrA = new ArrayList<Treenode>();
								if (tempptr.parentptrA == null) {
									// just single parent add to parent list
									newparentptrA.add(tempptr.parent);
									if (BDEBUG) {
										System.out.println("A " + tempptr.dmean
												+ " now links to "
												+ tempptr.parent.dmean);
									}
								} else {
									for (int i = 0; i < tempptr.parentptrA.length; i++) {
										// add all parent links to merging state
										newparentptrA
												.add(tempptr.parentptrA[i]);
										if (BDEBUG) {
											System.out
													.println("B "
															+ tempptr.dmean
															+ " now links to "
															+ tempptr.parentptrA[i].dmean);
										}
									}
								}

								bvalidmerge = true;

								if (BDEBUG) {
									if (tempptr.nextptr[0] != null)
										System.out.println("mean is "
												+ tempptr.nextptr[0].dmean);
									System.out.println("VALID MERGE");
								}
							}

							if (BDEBUG) {
								System.out.println(splitptr.dmean + " ");
							}

							if (splitptr.parentptrA == null) {
								splitptr.parent.nextptr[0] = tempptr;
								newparentptrA.add(splitptr.parent);
								if (BDEBUG) {
									System.out.println("C " + tempptr.dmean
											+ " now links to "
											+ splitptr.parent.dmean);
								}
							} else {
								for (int i = 0; i < splitptr.parentptrA.length; i++) {
									splitptr.parentptrA[i].nextptr[0] = tempptr;
									newparentptrA.add(splitptr.parentptrA[i]);
									if (BDEBUG) {
										System.out.println("D " + tempptr.dmean
												+ " now links to "
												+ splitptr.parentptrA[i].dmean);
									}
								}
							}

							if (tempptr.nextptr[0] != null) {
								tempptr.nextptr[0].parent = tempptr;
								tempptr.nextptr[0].parentptrA = null;
							}

							if (parentptr == splitptr.parent) {
								parentptr.numchildren--;
							}
						}
					}
				}

				if (bvalidmerge) {
					// copying over new parent nodes
					int nsize = newparentptrA.size();
					tempptr.parentptrA = new Treenode[nsize];
					for (int i = 0; i < nsize; i++) {
						tempptr.parentptrA[i] = newparentptrA.get(i);
					}
					if (BDEBUG) {
						System.out.println("MERGED TREE");
					}
					traverse(treeroot, 0, false);

					if (BVITERBI) {
						dlog = trainhmmV(treeroot, true);
					} else {
						dlog = trainhmm(treeroot, true);
					}

					if (BDEBUG) {
						System.out.println("after training:");
					}

					traverse(treeroot, 0, false);

					if (!bmodeldiff) {
						if (BVITERBI) {
							dlog = testhmmV(treeroot);
						} else {
							dlog = testhmm(treeroot);
						}
					}

					double dimprovemerge = (dlog - dprevouterbestlog)
							/ Math.abs(dlog);
					if (BDEBUG) {
						System.out.println("merge* " + dimprovemerge + "\t"
								+ MERGEMIN + "\t" + dlog + "\t"
								+ dprevouterbestlog + "\t" + dbestlog);
					}

					if (((!bmodeldiff) && (((dlog + MERGEMIN * Math.abs(dlog) - dprevouterbestlog) >= MERGEMINDIFF) && (dlog > dbestlog)))
							|| (bmodeldiff && (dlog >= dprevouterbestlog) && (dlog > dbestlog))) {
						bestTree = treeroot;
						computeNumLeaves(bestTree);
						dbestlog = dlog;
						if (BDEBUG) {
							System.out.println("in merge " + dbestlog
									+ " best tree is");
							traverse(treeroot, 0, false);
						}

						double dimprove = (dbestlog - dprevouterbestlog)
								/ Math.abs(dbestlog);
						String szimprove = nf3.format(100 * dimprove) + "%";
						String szimproveDIFF = nf3.format(dbestlog
								- dprevouterbestlog);
						if (statusLabel2 != null)
							statusLabel2.setText(" Merge change: "
									+ szimproveDIFF + " (" + szimprove + ")"
									+ " (Score " + nf2.format(dbestlog) + ")");
					}
				}
			}
		}
	}

	// /////////////////////////////////////////////////////////////////////
	/**
	 * A record containing a node id, mean, and standard deviation used for
	 * ordering the children node
	 */
	static class OrderRec {
		int nid;
		double dmean;
		double dsigma;

		OrderRec(double dmean, double dsigma, int nid) {
			this.nid = nid;
			this.dmean = dmean;
			this.dsigma = dsigma;
		}
	}

	/**
	 * Comparator for OrderRec sorts first on dmean, then dsigma, then id all in
	 * increasing order.
	 */
	static class OrderRecCompare implements Comparator<OrderRec> {
		public int compare(OrderRec or1, OrderRec or2) {
			if (or1.dmean < or2.dmean)
				return -1;
			else if (or1.dmean > or2.dmean)
				return 1;
			else if (or1.dsigma < or2.dsigma)
				return -1;
			else if (or1.dsigma > or2.dsigma)
				return 1;
			else if (or1.nid < or2.nid)
				return -1;
			else if (or1.nid > or2.nid)
				return 1;
			else
				return 0;
		}
	}

	// /////////////////////////////////////////////////////////////////////
	/**
	 * Sorts the children of each node in treeptr and descendants based on
	 * OrderRecCompare
	 */
	public void computeOrders(Treenode treeptr) {
		if (treeptr.numchildren >= 1) {
			if (treeptr.numchildren >= 2) {
				OrderRec[] theOrderRecs = new OrderRec[treeptr.numchildren];
				for (int nindex = 0; nindex < treeptr.numchildren; nindex++) {
					theOrderRecs[nindex] = new OrderRec(
							treeptr.nextptr[nindex].dmean,
							treeptr.nextptr[nindex].dsigma, nindex);
				}
				Arrays.sort(theOrderRecs, new OrderRecCompare());
				treeptr.orderA = new int[theOrderRecs.length];

				for (int nindex = 0; nindex < treeptr.orderA.length; nindex++) {
					treeptr.orderA[nindex] = theOrderRecs[nindex].nid;
				}
			}

			for (int nindex = 0; nindex < treeptr.numchildren; nindex++) {
				computeOrders(treeptr.nextptr[nindex]);
			}
		}
	}

	// /////////////////////////////////////////////////////////////////////

	/**
	 * Builds the initial tree which is just a single chain with mean and
	 * standard deviation for each node being the global mean and standard
	 * deviation at the time point
	 */
	public void buildEmptyTree(int ncurrdepth, Treenode currnode,
			double[] dsigma, double[] dmean) {
		currnode.dsigma = Math.max(Math.max(dsigma[ncurrdepth], dminstddev),
				MINSIGMA);
		if (ncurrdepth + 1 < dsigma.length) {
			currnode.numchildren = 1;
			currnode.nextptr[0] = new Treenode(currnode);
			currnode.nextptr[0].dmean = dmean[ncurrdepth];
			currnode.nextptr[0].parent = currnode;
			if (BREGDREM) {
				currnode.ptrans[0] = 1;
			}
			buildEmptyTree(ncurrdepth + 1, currnode.nextptr[0], dsigma, dmean);
		}
	}

	/**
	 * Initializes the model parameters to those in szinitfileval
	 */
	public void readInitTree(Treenode currnode) throws IOException {

		BufferedReader brinit = new BufferedReader(
				new FileReader(szinitfileval));
		String szfirstline = brinit.readLine();
		boolean bmergedtree = false;

		if (szfirstline.equalsIgnoreCase("MERGED TREE")) {
			szfirstline = brinit.readLine();
			bmergedtree = true;
		}

		if (bmergedtree) {
			if (ninitsearchval == 1) {
				throw new IllegalArgumentException(
						"Saved model has merges, but option allowed to start "
								+ "search from model is selected.\n"
								+ "Search may not begin from model with merges.  "
								+ "Consider option to use the saved model as is.");
			}

			if (!ballowmergeval) {
				throw new IllegalArgumentException(
						"Saved model has merges, but merges are not selected to be allowed.");
			}
		}

		StringTokenizer stfirstline = new StringTokenizer(szfirstline, "\t");
		if (stfirstline.countTokens() < 2) {
			throw new IllegalArgumentException("Not a valid saved model file!");
		}

		String szfirsttoken = stfirstline.nextToken();
		int numcoeff = Integer.parseInt(stfirstline.nextToken());
		if ((numcoeff != bindingData.regNames.length)
				&& (!szfirsttoken.startsWith("REGDREM"))) {
			throw new IllegalArgumentException(
					"Number of coefficients ("
							+ numcoeff
							+ ") in the saved model does not "
							+ "match the number of coefficients in the static input file ("
							+ bindingData.regNames.length + ")");
		}

		if (!(szfirsttoken.startsWith("REGDREM")) && (BREGDREM)) {
			throw new IllegalArgumentException(
					"'Use Transcription Factor-gene Interaction Data to Build Model' under 'Main Search Options' is unselected, "
							+ "but saved model was built using the data.");
		} else if ((szfirsttoken.startsWith("REGDREM")) && (!BREGDREM)) {
			throw new IllegalArgumentException(
					"'Use Transcription Factor-gene Interaction Data to Build Model' under 'Main Search Options' is selected, "
							+ "but saved model was built without the data.");
		}

		int ndepth = readInitTreeHelp(currnode, brinit, numcoeff, bmergedtree);

		if (ndepth != numcols) {
			throw new IllegalArgumentException(
					"Number of time points in the initial model ("
							+ ndepth
							+ ") does not match number of time points in the data ("
							+ numcols + ")");
		}

		String szLine = brinit.readLine();
		if ((szLine != null) && (szLine.equals("COLORS"))) {
			readColors(brinit);
		}

		if (bmergedtree) {
			HashMap<Integer, Treenode> createdNodes = new HashMap<Integer, Treenode>();
			mergeDuplicates(currnode, createdNodes);
		}
	}

	/**
	 * Helper function for reading the model parameters from a file
	 */
	private int readInitTreeHelp(Treenode currnode, BufferedReader brinit,
			int numcoeff, boolean bmergedtree) throws IOException {
		String szLine = brinit.readLine();
		StringTokenizer st = new StringTokenizer(szLine, "\t");
		int nID;

		if (bmergedtree) {
			nID = Integer.parseInt(st.nextToken());
			currnode.nID = nID;
		}

		currnode.dmean = Double.parseDouble(st.nextToken());
		currnode.dsigma = Double.parseDouble(st.nextToken());
		currnode.numchildren = Integer.parseInt(st.nextToken());

		int numcoefftoread;

		if (currnode.numchildren <= 1) {
			numcoefftoread = 0;
			currnode.binit = false;
		} else if (currnode.numchildren == 2) {
			numcoefftoread = numcoeff + 1;
			currnode.binit = true;
			numtotalPath++;
		} else {
			if (currnode.numchildren > nmaxchild) {
				throw new IllegalArgumentException(
						"Number of paths out of a node in the initial model file ("
								+ currnode.numchildren
								+ ") exceeds the maximum allowed " + nmaxchild);

			}
			numtotalPath += currnode.numchildren - 1;
			numcoefftoread = (numcoeff + 1) * (currnode.numchildren - 1);
			currnode.binit = true;
		}

		if ((numcoefftoread == 0) && (BREGDREM)) {
			currnode.ptrans[0] = 1;
		} else if (numcoefftoread > 0) {
			if (BREGDREM) {
				for (int nindex = 0; nindex < currnode.numchildren; nindex++) {
					currnode.ptrans[nindex] = Double.parseDouble(brinit
							.readLine());
				}
			} else {
				double[] dcoeff = new double[numcoefftoread];
				String szcoeff;
				StringTokenizer stcoeff;
				for (int ncoeffindex = 0; ncoeffindex < numcoefftoread; ncoeffindex++) {
					szcoeff = brinit.readLine();
					stcoeff = new StringTokenizer(szcoeff, "\t");
					String szcoeffname = stcoeff.nextToken();

					int nmodrow = ncoeffindex
							% (bindingData.regNames.length + 1);
					if ((nmodrow != 0)
							&& (!szcoeffname
									.equals(bindingData.regNames[nmodrow - 1]))) {
						throw new IllegalArgumentException(ncoeffindex + " "
								+ (bindingData.regNames.length + 1) + " "
								+ "Transcription factor " + szcoeffname
								+ " in saved model  "
								+ "does not match transcription factor "
								+ bindingData.regNames[nmodrow - 1] + ".");
					}
					dcoeff[ncoeffindex] = Double.parseDouble(stcoeff
							.nextToken());
				}

				currnode.tranC = new DREM_FastLogistic2(
						trainBinding.theInstancesIndex[currnode.ndepth][currnode.numchildren - 2],
						trainBinding.theInstances[currnode.ndepth][currnode.numchildren - 2],
						trainBinding.theInstancesRegIndex[currnode.ndepth][currnode.numchildren - 2],
						trainBinding.theInstancesReg[currnode.ndepth][currnode.numchildren - 2],
						trainBinding.ylabels[currnode.ndepth][currnode.numchildren - 2],
						currnode.numchildren, dcoeff, numbits,
						trainBinding.regTypes);
			}
		}

		if (currnode.numchildren == 0) {
			return 1;
		} else {
			int ndepth;
			currnode.nextptr[0] = new Treenode(currnode);
			currnode.nextptr[0].parent = currnode;
			ndepth = readInitTreeHelp(currnode.nextptr[0], brinit, numcoeff,
					bmergedtree);
			for (int nchild = 1; nchild < currnode.numchildren; nchild++) {
				currnode.nextptr[nchild] = new Treenode(currnode);
				currnode.nextptr[nchild].parent = currnode;
				if (ndepth != readInitTreeHelp(currnode.nextptr[nchild],
						brinit, numcoeff, bmergedtree)) {
					throw new IllegalArgumentException(
							"Invalid saved model file.  Two paths are not of the same length.");
				}
			}
			return (ndepth + 1);
		}
	}

	/**
	 * When reading a model with merges from a file some of the states can be
	 * listed more than once this procedures makes it so that in the internal
	 * representation the state only exists once
	 */
	private void mergeDuplicates(Treenode currnode,
			HashMap<Integer, Treenode> createdNodes) {
		if (currnode != null) {
			createdNodes.put(new Integer(currnode.nID), currnode);
			for (int nchild = 0; nchild < currnode.numchildren; nchild++) {
				Treenode tnode = createdNodes.get(new Integer(
						currnode.nextptr[nchild].nID));
				if (tnode == null) {
					mergeDuplicates(currnode.nextptr[nchild], createdNodes);
				} else {
					currnode.nextptr[nchild] = tnode;
					if (tnode.parentptrA == null) {
						tnode.parentptrA = new Treenode[2];
						tnode.parentptrA[0] = tnode.parent;
						tnode.parentptrA[1] = currnode;
					} else {
						Treenode[] temptr = tnode.parentptrA;
						tnode.parentptrA = new Treenode[tnode.parentptrA.length];
						for (int nindex = 0; nindex < temptr.length; nindex++) {
							tnode.parentptrA[nindex] = temptr[nindex];
						}
						tnode.parentptrA[temptr.length - 1] = currnode;
					}
				}
			}
		}
	}

	/**
	 * Loads saved colors for a DREM map
	 */
	public void readColors(BufferedReader brinit) throws IOException {
		savedColors = new ArrayList<Color>();
		String szLine;
		while ((szLine = brinit.readLine()) != null) {
			if (!szLine.trim().equals("")) {
				StringTokenizer st = new StringTokenizer(szLine, "\t");
				if (st.countTokens() != 4) {
					throw new IllegalArgumentException("A color line has "
							+ st.countTokens() + " values expecting 4");
				} else {
					float f1, f2, f3, f4;
					f1 = Float.parseFloat(st.nextToken());
					f2 = Float.parseFloat(st.nextToken());
					f3 = Float.parseFloat(st.nextToken());
					f4 = Float.parseFloat(st.nextToken());
					savedColors.add(new Color(f1, f2, f3, f4));
				}
			}
		}
	}

	// ////////////////////////////////////////////////////////////////////
	/**
	 * Helper function to traverseanddelayhelp that manipulates a split so that
	 * it occurs one time point later
	 */
	private Treenode delaysplit(int[] path, int ndesiredlevel, Treenode root,
			int nchildcombine) {
		Treenode treeroot = (Treenode) root.clone();
		Treenode ptr = treeroot;

		for (int nindex = 0; nindex < ndesiredlevel - 1; nindex++) {
			ptr = ptr.nextptr[path[nindex]];
		}
		Treenode removeptr = ptr.nextptr[path[ndesiredlevel - 1]];

		if (!removeptr.bchange) {
			return null;
		} else {
			for (int nj = path[ndesiredlevel - 1] + 1; nj < ptr.numchildren; nj++) {
				ptr.nextptr[nj - 1] = ptr.nextptr[nj];
			}

			ptr.numchildren--;
			ptr.binit = false;

			if (BREGDREM) {
				double dsum = 0;
				for (int nindex = 0; nindex < ptr.numchildren; nindex++) {
					dsum += ptr.ptrans[nindex];
				}

				for (int nindex = 0; nindex < ptr.numchildren; nindex++) {
					ptr.ptrans[nindex] /= dsum;
				}
			}

			int ncurrnumchild = ptr.nextptr[nchildcombine].numchildren;

			Treenode combineptr = ptr.nextptr[nchildcombine];
			for (int nindex = 0; nindex < removeptr.numchildren; nindex++) {
				combineptr.nextptr[nindex + ncurrnumchild] = removeptr.nextptr[nindex];
				removeptr.nextptr[nindex].parent = combineptr;
			}

			if (BREGDREM) {
				for (int nindex = 0; nindex < combineptr.numchildren; nindex++) {
					combineptr.ptrans[nindex] = combineptr.ptrans[nindex] * 0.5;
				}

				for (int nindex = 0; nindex < removeptr.numchildren; nindex++) {
					combineptr.ptrans[nindex + combineptr.numchildren] = removeptr.ptrans[nindex] * 0.5;
				}
			}
			combineptr.numchildren += removeptr.numchildren;
			combineptr.binit = false;
		}
		return treeroot;
	}

	// ////////////////////////////////////////////////////////////////////
	/**
	 * Deletes a child from the specified path on a cloned version of the root
	 * node
	 */
	public Treenode deletepath(int[] path, int ndesiredlevel, Treenode root) {
		// used to be call deleteleaf
		Treenode treeroot = (Treenode) root.clone();
		Treenode ptr = treeroot;

		for (int nindex = 0; nindex < ndesiredlevel - 1; nindex++) {
			ptr = ptr.nextptr[path[nindex]];
		}

		if (!ptr.nextptr[path[ndesiredlevel - 1]].bchange) {
			return null;
		} else {
			for (int nj = path[ndesiredlevel - 1] + 1; nj < ptr.numchildren; nj++) {
				ptr.nextptr[nj - 1] = ptr.nextptr[nj];
			}

			ptr.numchildren--;
			ptr.binit = false;

			if (BREGDREM) {
				double dsum = 0;
				for (int nindex = 0; nindex < ptr.numchildren; nindex++) {
					dsum += ptr.ptrans[nindex];
				}

				for (int nindex = 0; nindex < ptr.numchildren; nindex++) {
					ptr.ptrans[nindex] /= dsum;
				}
			}
		}
		return treeroot;
	}

	// //////////////////////////////////////////////////////////////////////
	/**
	 * With its helper functions adds the specified path to the model
	 */
	public void traverseandadd(int[] path, Treenode ptr, Treenode origroot)
			throws Exception {
		traverseandaddhelp(path, 0, ptr, origroot);
	}

	// ////////////////////////////////////////////////////////////////////

	/**
	 * Helper function for traverseandaddhelp recursively tries to add paths to
	 * the model, and then evaluates the improvement
	 */
	private void traverseandaddhelp(int[] path, int nlevel, Treenode ptr,
			Treenode origroot) throws Exception {
		// searches for the best place to add a node at current level
		// tries adding path at current level
		if ((nlevel < path.length) && (!DREM_IO.bendsearch)) {
			for (int nindex = 0; nindex < ptr.numchildren; nindex++) {
				path[nlevel] = nindex;
				traverseandaddhelp(path, nlevel + 1, ptr.nextptr[nindex],
						origroot);
			}
		}

		if (((ptr != null) && ((ptr.numchildren < nmaxchild)))
				&& (!DREM_IO.bendsearch)) {
			if (DREM_IO.bdisplaycurrent) {
				displayTempMap();
			}

			if (nlevel < path.length) {
				// mark to try to split
				if (BDEBUG) {
					for (int ni = 0; ni < nlevel; ni++) {
						System.out.print(path[ni] + "\t");
					}
				}

				Treenode splittree = splitnode(path, nlevel, origroot);

				if (BDEBUG) {
					System.out.println(path[nlevel]);
				}

				if (splittree != null) {
					traverse(splittree, 0, false);

					double dlog;
					if (BVITERBI) {
						dlog = trainhmmV(splittree, false);
					} else {
						dlog = trainhmm(splittree, false);
					}

					if (!bmodeldiff) {
						if (BVITERBI) {
							dlog = testhmmV(splittree);
						} else {
							dlog = testhmm(splittree);
						}
					}
					traverse(splittree, 0, false);

					if (BDEBUG) {
						System.out.println("*test " + dlog);
					}

					if ((dlog > dbestlog)
							&& (bmodeldiff || (dlog - dprevouterbestlog) > EPSILONDIFF)) {
						bestTree = splittree;
						dbestlog = dlog;
						dbesttrainlike = dtrainlike;
						double dimprove = (dbestlog - dprevouterbestlog)
								/ Math.abs(dbestlog);
						String szimprove = nf3.format(100 * dimprove) + "%";
						String szimproveDIFF = nf3.format(dbestlog
								- dprevouterbestlog);

						if (statusLabel3 != null) {
							statusLabel3.setText(" Next score: "
									+ nf2.format(dbestlog));
						}

						if (statusLabel2 != null) {
							statusLabel2.setText(" Next score improvement: "
									+ szimproveDIFF + " (" + szimprove + ")");
						}
					}
				}
			}
		}
	}

	// ///////////////////////////////////////////////////////////////////
	/**
	 * Helper function for adding a path in traverseandaddhelp. Responsible for
	 * splitting one path out of a node into two.
	 */
	private Treenode splitnode(int[] path, int ndesiredlevel, Treenode origroot) {
		Treenode treeroot = (Treenode) origroot.clone();
		Treenode ptr = treeroot;

		for (int nindex = 0; nindex < ndesiredlevel; nindex++) {
			// advancing pointer to the node from which a new path should be
			// added
			ptr = ptr.nextptr[path[nindex]];
		}

		if (!ptr.bchange) {
			return null;
		} else {
			// increasing the number of children
			ptr.numchildren++;
			ptr.binit = false;

			// readjusting the transition probabilities
			// //////////////////////////////////////

			if (BREGDREM) {
				for (int nindex = 0; nindex < ptr.numchildren - 1; nindex++) {

					ptr.ptrans[nindex] = ptr.ptrans[nindex]
							* (ptr.numchildren - 1) / (ptr.numchildren);
				}
				ptr.ptrans[ptr.numchildren - 1] = 1.0 / (ptr.numchildren);

				if (BDEBUG) {
					System.out.println("probs A " + ptr.ptrans[0] + "\t"
							+ ptr.ptrans[1]);
				}
			}

			if (ndesiredlevel < path.length) {
				path[ndesiredlevel] = ptr.numchildren - 1;
			}
			// //////////////////////////////////////////////////
			// more sophisticated initialization schemes possible here
			// /////////////////////////////////////
			for (int nlevel = ndesiredlevel; nlevel < path.length; nlevel++) {
				ptr.nextptr[ptr.numchildren - 1] = new Treenode(ptr);
				if (ptr.numchildren > 1) {
					double dk = 0.02;
					double ddiff = ptr.dmean
							- ptr.nextptr[ptr.numchildren - 2].dmean;

					if (Math.abs(ddiff) < dk * ptr.dsigma) {
						if (ptr.nextptr[ptr.numchildren - 2].dmean > ptr.dmean) {
							ptr.nextptr[ptr.numchildren - 1].dmean = ptr.dmean
									- (dk * ptr.dsigma - (ptr.nextptr[ptr.numchildren - 2].dmean - ptr.dmean));

							if (BDEBUGMODEL) {
								System.out
										.println("A: MEAN "
												+ ptr.nextptr[ptr.numchildren - 1].dmean
												+ " "
												+ ptr.nextptr[ptr.numchildren - 2].dmean
												+ " " + ptr.dsigma);
							}
						} else {
							ptr.nextptr[ptr.numchildren - 1].dmean = ptr.dmean
									+ (dk * ptr.dsigma - (ptr.dmean - ptr.nextptr[ptr.numchildren - 2].dmean));

							if (BDEBUGMODEL) {
								System.out
										.println("B: MEAN "
												+ ptr.nextptr[ptr.numchildren - 1].dmean
												+ " "
												+ ptr.nextptr[ptr.numchildren - 2].dmean
												+ " " + ptr.dsigma);
							}
						}
					} else {
						ptr.nextptr[ptr.numchildren - 1].dmean = ptr.dmean;

						if (BDEBUGMODEL) {
							System.out.println("C: MEAN "
									+ ptr.nextptr[ptr.numchildren - 1].dmean
									+ " "
									+ ptr.nextptr[ptr.numchildren - 2].dmean
									+ " " + ptr.dsigma);
						}
					}
				} else {
					ptr.nextptr[ptr.numchildren - 1].dmean = ptr.dmean;
				}

				ptr.nextptr[ptr.numchildren - 1].dsigma = ptr.dsigma;

				ptr = ptr.nextptr[ptr.numchildren - 1];
				ptr.numchildren = 1;
				if (BREGDREM) {
					ptr.ptrans[0] = 1;
				}
			}
			ptr.numchildren = 0;
		}
		return treeroot;
	}

	// ////////////////////////////////////////////////////////////////////
	/**
	 * With its helper function finds the path with the fewest genes assigned
	 * this will become a candidate for deletion
	 */
	public MinPathRec traverseanddeleteMinPath(int[] path, int[] bestpath,
			Treenode ptr) throws Exception {
		return traverseanddeletehelpMinPath(path, 0, ptr);
	}

	// ////////////////////////////////////////////////////////////////////
	/**
	 * A record used for storing the path with the fewest genes assigned
	 */
	static class MinPathRec {
		int nval;
		int nlevel;
		int[] bestpath;

		MinPathRec(int nval, int nlevel, int[] bestpath) {
			this.nval = nval;
			this.nlevel = nlevel;
			this.bestpath = bestpath;
		}
	}

	/**
	 * Deletes the specificed path from the model starting from ndesiredlevel
	 */
	public void deleteMinPath(int[] path, int ndesiredlevel, Treenode root) {
		Treenode ptr = root;

		for (int nindex = 0; nindex < ndesiredlevel - 1; nindex++) {
			ptr = ptr.nextptr[path[nindex]];
		}

		for (int nj = path[ndesiredlevel - 1] + 1; nj < ptr.numchildren; nj++) {
			ptr.nextptr[nj - 1] = ptr.nextptr[nj];
		}

		ptr.numchildren--;
		ptr.binit = false;

		if (BREGDREM) {
			double dsum = 0;
			for (int nindex = 0; nindex < ptr.numchildren; nindex++) {
				dsum += ptr.ptrans[nindex];
			}

			for (int nindex = 0; nindex < ptr.numchildren; nindex++) {
				if (dsum > 0) {
					ptr.ptrans[nindex] /= dsum;
				} else {
					ptr.ptrans[nindex] = 0;
				}
			}
		}
	}

	/**
	 * Recursively searches for the path with the fewest genes assigned this
	 * will become a candidate for deletion
	 */
	private MinPathRec traverseanddeletehelpMinPath(int[] path, int nlevel,
			Treenode ptr) throws Exception {
		int nmin = Integer.MAX_VALUE;
		if (nlevel < path.length) {
			MinPathRec theminMinPathRec = null;
			for (int nindex = 0; nindex < ptr.numchildren; nindex++) {
				path[nlevel] = nindex;
				MinPathRec thetempMinPathRec = traverseanddeletehelpMinPath(
						path, nlevel + 1, ptr.nextptr[nindex]);

				if (thetempMinPathRec.nval < nmin) {
					nmin = thetempMinPathRec.nval;
					theminMinPathRec = thetempMinPathRec;
					theminMinPathRec.bestpath[nlevel] = nindex;
					if (thetempMinPathRec.nval == ptr.numPath) {
						thetempMinPathRec.nlevel = nlevel;
					}
				}
			}
			return theminMinPathRec;
		} else {
			MinPathRec theMinPathRec = new MinPathRec(ptr.numPath, nlevel,
					new int[path.length]);
			return theMinPathRec;
		}
	}

	// //////////////////////////////////////////////////////////////////////////
	/**
	 * With its helper method traverseanddelayhelp searches for splits to delay
	 * one time point Returns true if an eligible split to delay was found
	 */
	public boolean traverseanddelay(int[] path, Treenode ptr,
			int ndesiredlevel, Treenode origroot, boolean bresplit)
			throws Exception {
		if (BDEBUG) {
			System.out.println("entering traverse and delay");
		}
		return traverseanddelayhelp(path, 0, ndesiredlevel, ptr, origroot,
				bresplit, 0);
	}

	// ///////////////////////////////////////////////////////////////////////////
	/**
	 * Recursively searches for splits to delay one time point
	 */
	private boolean traverseanddelayhelp(int[] path, int nlevel,
			int ndesiredlevel, Treenode ptr, Treenode origroot,
			boolean bresplit, int nchild) throws Exception {

		if (BDEBUG) {
			System.out.println("traverseanddelayhelp " + ndesiredlevel + " "
					+ ndesiredlevel);
		}

		boolean breturnval = false;
		if ((nlevel < path.length - 1) && (nlevel < ndesiredlevel)) {
			for (int nindex = 0; nindex < ptr.numchildren; nindex++) {
				if (DREM_IO.bdisplaycurrent) {
					displayTempMap();
				}
				path[nlevel] = nindex;
				boolean breturnvaltemp;
				breturnvaltemp = traverseanddelayhelp(path, nlevel + 1,
						ndesiredlevel, ptr.nextptr[nindex], origroot, bresplit,
						nindex);
				breturnval = breturnval || breturnvaltemp;
			}
		}

		// assuming ptr.parent also not null
		if ((ptr != null) && (nchild >= 1) && (nlevel == ndesiredlevel)) {
			for (int nchildcombine = 0; nchildcombine < nchild; nchildcombine++) {
				if (ptr.numchildren
						+ ptr.parent.nextptr[nchildcombine].numchildren <= nmaxchild) {
					double dabsmeandiff = Math.abs(ptr.dmean
							- ptr.parent.nextptr[nchildcombine].dmean);
					NumberFormat nf4 = NumberFormat.getInstance();
					nf4.setMinimumFractionDigits(4);
					nf4.setMaximumFractionDigits(4);
					if (BDEBUG) {
						System.out
								.println(nlevel
										+ " "
										+ nf4.format(ptr.dmean)
										+ " "
										+ nf4
												.format(ptr.parent.nextptr[nchildcombine].dmean)
										+ " "
										+ nf4.format(dabsmeandiff)
										+ " "
										+ nf4.format(ptr.dsigma)
										+ " "
										+ nf4
												.format(ptr.parent.nextptr[nchildcombine].dsigma));
						System.out.println("-----------");
					}

					// mark to try to delete
					if (BDEBUG) {
						System.out.print("dy ");
						for (int ni = 0; ni < nlevel; ni++) {
							System.out.print(path[ni] + " ");
						}
						System.out.println();
						System.out.println("before");
					}

					traverse(origroot, 0, false);
					Treenode delaytree = delaysplit(path, nlevel, origroot,
							nchildcombine);

					if (delaytree != null) {
						double dlog;
						if (BVITERBI) {
							dlog = trainhmmV(delaytree, true);
						} else {
							dlog = trainhmm(delaytree, true);
						}

						if (!bmodeldiff) {
							if (BVITERBI) {
								dlog = testhmmV(delaytree);
							} else {
								dlog = testhmm(delaytree);
							}
						}

						if (BDEBUG) {
							System.out.println("after " + dlog);
						}
						traverse(delaytree, 0, false);

						if (BDEBUG) {
							System.out.println("test " + dlog);
						}

						double ddelayimprove = (dlog - dprevouterbestlog)
								/ Math.abs(dlog);

						if (BDEBUG) {
							System.out.println("DELAY\t" + dlog + "\t"
									+ dbestlog + "\t" + ddelayimprove + "\t"
									+ (dlog - dprevouterbestlog)
									/ Math.abs(dlog) + "\t" + DELAYMIN);
						}

						if (BDEBUG) {
							System.out.println("delay = " + ddelayimprove
									+ "\t" + DELAYMIN);
						}

						if ((((dlog + DELAYMIN * Math.abs(dlog) - dprevouterbestlog) >= DELAYMINDIFF)
								&& (dlog > dbestlog) && !bmodeldiff)
								|| ((bmodeldiff) && (dlog >= dprevouterbestlog) && (dlog > dbestlog))) {
							bestTree = delaytree;
							dbestlog = dlog;
							if (BDEBUG) {
								System.out.println("in delay " + dbestlog);
							}

							double dimprove = (dbestlog - dprevouterbestlog)
									/ Math.abs(dbestlog);
							String szimprove = nf3.format(100 * dimprove) + "%";
							String szimproveDIFF = nf3.format(dbestlog
									- dprevouterbestlog);
							if (statusLabel2 != null) {
								statusLabel2.setText(" Delay next change: "
										+ szimproveDIFF + " (" + szimprove
										+ ")" + " (Score "
										+ nf2.format(dbestlog) + ")");

							}

							dbesttrainlike = dtrainlike;
							breturnval = true;
						}
					}
				}
			}
		}
		return breturnval;
	}

	// ////////////////////////////////////////////////////////////////////
	/**
	 * With its helper function searches for the best path to delete Returns
	 * true if an eligible path to delete was found
	 */
	public boolean traverseanddelete(int[] path, Treenode ptr,
			Treenode origroot, boolean bresplit, double dimprovemin,
			double dimprovemindiff) throws Exception {
		return traverseanddeletehelp(path, 0, ptr, origroot, bresplit,
				dimprovemin, dimprovemindiff);
	}

	// ////////////////////////////////////////////////////////////////////
	/**
	 * Helper function that searches for the best path to delete
	 */
	private boolean traverseanddeletehelp(int[] path, int nlevel, Treenode ptr,
			Treenode origroot, boolean bresplit, double dimprovemin,
			double dimprovemindiff) throws Exception {

		boolean breturnval = false;
		if (nlevel < path.length) {
			for (int nindex = 0; nindex < ptr.numchildren; nindex++) {
				if (DREM_IO.bdisplaycurrent) {
					displayTempMap();
				}

				path[nlevel] = nindex;
				boolean breturnvaltemp = traverseanddeletehelp(path,
						nlevel + 1, ptr.nextptr[nindex], origroot, bresplit,
						dimprovemin, dimprovemindiff);
				breturnval = (breturnvaltemp || breturnval);
			}
		}

		if ((ptr != null) && (ptr.parent != null)
				&& (ptr.parent.numchildren > 1)) {
			if (BDEBUG) {
				// mark to try to delete
				for (int ni = 0; ni < nlevel; ni++) {
					System.out.print(path[ni] + " ");
				}
				System.out.println();
			}

			Treenode deletetree = deletepath(path, nlevel, origroot);

			if (deletetree != null) {
				double dlog;

				if (BVITERBI) {
					dlog = trainhmmV(deletetree, false);
				} else {
					dlog = trainhmm(deletetree, false);
				}

				if (!bmodeldiff) {
					if (BVITERBI) {
						dlog = testhmmV(deletetree);
					} else {
						dlog = testhmm(deletetree);
					}
				}

				if (BDEBUG) {
					System.out.println("test " + dlog);
				}

				double dimprovedelete = (dlog - dprevouterbestlog)
						/ Math.abs(dlog);
				if (BDEBUG) {
					System.out.println("del* " + dimprovedelete + "\t"
							+ dimprovemin + "\t" + (dlog - dprevouterbestlog));
					System.out.println("~~" + dlog + "\t" + dbestlog + "\t"
							+ dprevouterbestlog);

					System.out.println("DELETE\t" + dlog + "\t" + dbestlog
							+ "\t" + dimprovedelete + "\t"
							+ (dlog - dprevouterbestlog) / Math.abs(dlog)
							+ "\t" + dimprovemin);
				}

				if ((((dlog + dimprovemin * Math.abs(dlog) - dprevouterbestlog) >= dimprovemindiff)
						&& (dlog > dbestlog) && !bmodeldiff)
						|| (bmodeldiff && (dimprovemin >= dprevouterbestlog) && (dlog > dbestlog)))

				{
					bestTree = deletetree;
					dbestlog = dlog;
					if (BDEBUG) {
						System.out.println("in delete " + dbestlog);
					}

					if (!bresplit) {
						double dimprove = (dbestlog - dprevouterbestlog)
								/ Math.abs(dbestlog);
						String szimprove = nf3.format(100 * dimprove) + "%";
						String szimproveDIFF = nf3
								.format((dbestlog - dprevouterbestlog));
						if (statusLabel3 != null) {
							statusLabel3.setText(" Next score: "
									+ nf2.format(dbestlog));
						}

						if (statusLabel2 != null) {
							statusLabel2.setText(" Next score improvement: "
									+ szimproveDIFF + " (" + szimprove + ")");
						}
					} else {
						double dimprove = (dbestlog - dprevouterbestlog)
								/ Math.abs(dbestlog);
						String szimprove = nf3.format(100 * dimprove) + "%";
						String szimproveDIFF = nf3
								.format((dbestlog - dprevouterbestlog));

						if (statusLabel2 != null) {
							statusLabel2.setText(" Delete path change : "
									+ szimproveDIFF + " (" + szimprove + ")");
						}
					}
					dbesttrainlike = dtrainlike;
					breturnval = true;
				}
			}
		}

		return breturnval;
	}

	// /////////////////////////////////////////////////////////////////////
	/**
	 * Prints the most significant TF-association scores for each TF
	 */
	private void printMinPvals() {
		double[] bestpsplit = new double[bindingData.regNames.length];
		double[] bestpfull = new double[bindingData.regNames.length];

		int[] bestsplitdepth = new int[bindingData.regNames.length];
		int[] bestfulldepth = new int[bindingData.regNames.length];

		for (int ni = 0; ni < bestpsplit.length; ni++) {
			bestpsplit[ni] = 1;
		}

		for (int ni = 0; ni < bestpsplit.length; ni++) {
			bestpfull[ni] = 1;
		}

		getMinPvals(treeptr, bestpsplit, bestpfull, bestsplitdepth,
				bestfulldepth, 1);

		try {
			PrintWriter pw = new PrintWriter(new FileOutputStream(SCORESDIR
					+ "/" + nrandomseed + ".txt"));
			for (int ni = 0; ni < bindingData.regNames.length; ni++) {
				pw.println(bindingData.regNames[ni] + "\t" + bestpfull[ni]
						+ "\t" + bestpsplit[ni] + "\t" + bestfulldepth[ni]
						+ "\t" + bestsplitdepth[ni]);
			}
			pw.close();
		} catch (IOException ioex) {
			ioex.printStackTrace(System.out);
		}
	}

	/**
	 * Helper procedure for getting the most significant TF-association scores
	 * for each TF
	 */
	private void getMinPvals(Treenode ptr, double[] bestpsplit,
			double[] bestpfull, int[] bestsplitdepth, int[] bestfulldepth,
			int ndepth) {
		if (ptr.numchildren >= 1) {
			for (int nchild = 0; nchild < ptr.numchildren; nchild++) {
				if (ptr.dpvalEdgeSplit != null) {
					for (int ni = 0; ni < bestpsplit.length; ni++) {
						if (ptr.dpvalEdgeSplit[nchild][ni] < bestpsplit[ni]) {
							bestpsplit[ni] = ptr.dpvalEdgeSplit[nchild][ni];
							bestsplitdepth[ni] = ndepth;
						}
					}
				}

				if (ptr.dpvalEdgeFull.length >= 1) {
					for (int ni = 0; ni < bestpfull.length; ni++) {
						if (ptr.dpvalEdgeFull[nchild][ni] < bestpfull[ni]) {
							bestpfull[ni] = ptr.dpvalEdgeFull[nchild][ni];
							bestfulldepth[ni] = ndepth;
						}
					}
				}

				getMinPvals(ptr.nextptr[nchild], bestpsplit, bestpfull,
						bestsplitdepth, bestfulldepth, ndepth + 1);
			}
		}
	}

	// ////////////////////////////////////////////////////////////////////
	/**
	 * A Treenode corresponds to a state in the model
	 */
	class Treenode {
		int ncurrtime;// use to count visited
		int nID;
		int niteration;
		int ninstanceiteration;
		int nrec;
		int ninstancerec;
		int nminparentlevel;
		Treenode minparentptr;

		Treenode[] parentptrA;
		/** number of leaves descendants of this node, only relevant if split */
		int numdownleaves;
		double dens;
		/** expected sum of the emissions from this node */
		double dEsum;
		/** expected sum of the square of emissions from this node */
		double dEsumsq;
		double dPsum;
		/** emission mean */
		double dmean;
		/** emission std */
		double dsigma;
		double dpredictweight;
		String szgolabel = "";
		String szgenesetlabel = "";
		String sztfsetlabel = "";
		/** non zero recs */
		int nprime;
		/** number of children */
		int numchildren;
		/** can change state's values */
		boolean bchange;
		int ndepth;
		int nparentcolorindex;
		boolean[] bvector;

		/** The logistic regression classifier */
		DREM_FastLogistic2 tranC;
		double[] recweight;
		double[] dTotalScore;
		double[][] dpvals;
		int[] orderA;
		/**
		 * Stores the number of genes regulated by a TF on a given path with the
		 * given interaction value. Dimensions: <br>
		 * [num TFs][num paths out of split][num distinct binding interaction
		 * values]
		 */
		int[][][] ncountvals;
		/**
		 * Stores the number of genes regulated by a TF on the paths other than
		 * the given path with the given interaction value. Dimensions: <br>
		 * [num TFs][num paths out of split][num distinct binding interaction
		 * values]
		 */
		int[][][] nothersum;
		double[][] dscoreother;
		double[][] dscore;
		double[][] ddiff;
		/**
		 * A TreeSet used to display important TFs based on activity score,
		 * which isn't a measure of significance in the statistical sense.
		 */
		TreeSet<SigTFRecv2>[] tsSigTFActivity;
		TreeSet<SigTFRecv2> tsSigTF;
		TreeSet<SigTFRecv2>[] tsSigTFEdgeSplit;
		TreeSet<SigTFRecv2>[] tsSigTFFull;
		Hashtable<Double, Integer>[] htScore;
		/**
		 * Stores the number of genes regulated by a TF on all paths out of a
		 * split with the given interaction value. Dimensions: <br>
		 * [num TFs][num distinct binding interaction values]
		 */
		int[][] ncountTotals;
		double[][] dpvalEdgeSplit;
		double[][] ddiffEdgeSplit;
		double[][] dpvalEdgeFull;
		double[][] ddiffEdgeFull;
		double[][] dexpectEdgeFull;
		double[][] dexpectEdgeSplit;
		double[][] dratioSplit;
		/**
		 * An array of activity scores for all TFs based on how the genes the TF
		 * regulates transition to the next states compared to how all genes
		 * into the split transistion.
		 */
		double[] dTFActivityScore;

		boolean[] bInNode;
		/** Number of genes going through this state??? */
		int numPath;
		/** probability of transitioning to each of the next states */
		double[] ptrans;
		/** expected number of transitions given the training sequences */
		double[] aprobs;
		/** pointer to each of the next states */
		Treenode[] nextptr;
		/** pointer back to the parent */
		Treenode parent;

		PText thepredictText;
		PText goText;
		PText genesetText;
		PText tfsetText;
		/**
		 * forward variable for current variables <br>
		 * p(x_1,...,x_i,pi_i=k)
		 */
		double df;
		/**
		 * backward variable for current variables <br>
		 * p(x_(i+1),...,x_L|pi_i=k)
		 */
		double db;
		boolean binit;

		/**
		 * Calls inittreenode with a null parent
		 */
		Treenode() {
			inittreenode(null);
		}

		Treenode(Treenode parent) {
			inittreenode(parent);
		}

		/**
		 * Calls inittreenode with parent
		 */
		void inittreenode(Treenode parent) {
			niteration = 0;
			ninstanceiteration = 0;
			nrec = -1;
			dmean = 0;
			dsigma = .5;
			this.parent = parent;
			dEsum = 0;
			dEsumsq = 0;
			dPsum = 0;
			nprime = 0;
			bchange = true;
			binit = false;
			if (parent == null) {
				ndepth = 0;
			} else {
				ndepth = parent.ndepth + 1;
			}
			nextptr = new Treenode[nmaxchild];

			if (BREGDREM) {
				ptrans = new double[nmaxchild];
				aprobs = new double[nmaxchild];
			}

			bvector = new boolean[numbits];

			if (bindingData.gene2RegBinding != null) {
				dTotalScore = new double[numbits];
			}

			for (int nindex = 0; nindex < numbits; nindex++) {
				bvector[nindex] = false;
			}

			numchildren = 0;
			for (int nindex = 0; nindex < nextptr.length; nindex++) {
				nextptr[nindex] = null;
			}
		}

		/**
		 * For making copy of nodes
		 */
		public Object clone() {
			if (parent == null) {
				if (BDEBUG) {
					System.out.println("in clone........");
				}
				htBackPts = new Hashtable<Treenode, BackPtrRec>();
			}

			Treenode tnode = new Treenode();
			tnode.nID = nID;
			tnode.dmean = dmean;
			tnode.dsigma = dsigma;

			tnode.ndepth = ndepth;
			tnode.dEsum = dEsum;
			tnode.dEsumsq = dEsumsq;
			tnode.dPsum = dPsum;
			tnode.nprime = nprime;
			tnode.numchildren = numchildren;

			tnode.thepredictText = thepredictText;
			tnode.parent = null;
			tnode.bchange = bchange;
			tnode.df = df;
			tnode.db = db;
			tnode.binit = binit;
			tnode.nextptr = new Treenode[nmaxchild];

			if (BREGDREM) {
				tnode.ptrans = new double[nmaxchild];
				tnode.aprobs = new double[nmaxchild];
			} else if (tranC != null) {
				tnode.tranC = (DREM_FastLogistic2) tranC.clone(); // do we need
				// a clone
				// yes
			}

			tnode.recweight = new double[recweight.length];

			for (int nindex = 0; nindex < nextptr.length; nindex++) {
				if (nextptr[nindex] == null) {
					tnode.nextptr[nindex] = null;
				} else {
					BackPtrRec childNodeRec = htBackPts.get(nextptr[nindex]);

					if (childNodeRec != null) {
						// we've already cloned the child node, just have it
						// link to it
						tnode.nextptr[nindex] = childNodeRec.childNode;
						// have the child link back to this node
						if (BDEBUG) {
							System.out.println("[[" + childNodeRec + " "
									+ childNodeRec.childNode + " "
									+ childNodeRec.childNode.parentptrA);
						}

						int nfindparentindex = 0;
						while (nextptr[nindex].parentptrA[nfindparentindex] != this) {
							nfindparentindex++;
						}
						if (BDEBUG) {
							System.out.println("nfindparentindex = "
									+ nfindparentindex);
						}
						childNodeRec.childNode.parentptrA[nfindparentindex] = tnode;
					} else {
						// cloning a new node
						tnode.nextptr[nindex] = (Treenode) nextptr[nindex]
								.clone();
						tnode.nextptr[nindex].parent = tnode;

						if (nextptr[nindex].parentptrA != null) {
							tnode.nextptr[nindex].parentptrA = new Treenode[nextptr[nindex].parentptrA.length];
							for (int nparentindex = 0; nparentindex < nextptr[nindex].parentptrA.length; nparentindex++) {
								if (nextptr[nindex].parentptrA[nparentindex] == this) {
									if (BDEBUG) {
										System.out.println("~~``"
												+ tnode.nextptr[nindex].dmean
												+ " " + nparentindex);
									}
									tnode.nextptr[nindex].parentptrA[nparentindex] = tnode;
								}
							}

							if (nextptr[nindex].parentptrA.length >= 2) {
								if (BDEBUG) {
									System.out.println(dmean + " "
											+ nextptr[nindex].dmean);
								}
								htBackPts.put(nextptr[nindex], new BackPtrRec(
										tnode.nextptr[nindex]));
							}
						}
					}
				}

				if (BREGDREM) {
					tnode.ptrans[nindex] = ptrans[nindex];
					tnode.aprobs[nindex] = aprobs[nindex];
				}
			}

			tnode.bvector = new boolean[numbits];
			for (int nindex = 0; nindex < numbits; nindex++) {
				tnode.bvector[nindex] = bvector[nindex];
			}

			return tnode;
		}

		/**
		 * 
		 * @return a map from TF names (excluding the " [#]" used to indicate
		 *         the primary path out of the split for the TF) to the best
		 *         activity score seen at any of this node's ancestors.
		 */
		public HashMap<String, Double> getAncestorActivityScores() {
			HashMap<String, Double> bestSeenScores = new HashMap<String, Double>();
			// TODO need to handle the case where there are multiple parents?
			if (parent != null) {
				return parent.getAncestorActivityScores(bestSeenScores);
			} else {
				return bestSeenScores;
			}
		}

		/**
		 * Helper function for the other version of getAncestorActivityScores.
		 * 
		 * @param bestSeenScores
		 *            The best activity scores seen in ancestors of the original
		 *            node before the search reached the current node.
		 * @return a map from TF names (excluding the " [#]" used to indicate
		 *         the primary path out of the split for the TF) to the best
		 *         activity score seen at any of this nodes ancestors or in the
		 *         bestSeenScores.
		 */
		private HashMap<String, Double> getAncestorActivityScores(
				HashMap<String, Double> bestSeenScores) {
			// Enumerate all TFs in the tree set and add their activty
			// score to the map if there was no score for that TF or if
			// the score at the current node is better (i.e. lower, because
			// the activity scores have been transformed to resemble p-values)
			for(int i = 0; i < numchildren; i++)
			{
				Iterator<DREM_Timeiohmm.SigTFRecv2> itr = tsSigTFActivity[i]
						.iterator();
				DREM_Timeiohmm.SigTFRecv2 theTFRec;
				while (itr.hasNext()) {
					theTFRec = itr.next();
	
					String tfName = theTFRec.szname;
					// Check if the path out of the split has been added as " [#]"
					// to the TF name, and if so remove it
					// need to be done
//					if (tfName.endsWith("]")) {
//						tfName = tfName.substring(0, tfName.lastIndexOf(" ["));
//					}
	
					// Add the TF and score to the map if they are not already
					// present
					if (!bestSeenScores.containsKey(tfName)) {
						// The "pval" is actually the transformed activity score
						bestSeenScores.put(tfName, Double.valueOf(theTFRec.dpval));
					}
					// Otherwise add them if the current score is lower
					// than the previously seen best lower score
					else {
						double bestScore = ((Double) bestSeenScores.get(tfName))
								.doubleValue();
	
						if (theTFRec.dpval < bestScore) {
							bestSeenScores.put(tfName, Double
									.valueOf(theTFRec.dpval));
						}
					}
				} // end iteration over all TF activity scores at this node
			}
			if (parent != null) {
				bestSeenScores = parent
						.getAncestorActivityScores(bestSeenScores);
			}

			return bestSeenScores;
		}
	}

	/**
	 * A record with information about a state a back pointer points to
	 */
	static class BackPtrRec {
		BackPtrRec(Treenode childNode) {
			this.childNode = childNode;
		}

		Treenode childNode;
	}

	// ///////////////////////////////////////////////////////////////////
	/**
	 * Returns true iff treeptr or a descendent has two or more parents
	 */
	public boolean hasMerge(Treenode treeptr) {
		boolean bhasmerge = ((treeptr.parentptrA != null) && (treeptr.parentptrA.length >= 2));
		for (int nchild = 0; (nchild < treeptr.numchildren) && (!bhasmerge); nchild++) {
			bhasmerge = hasMerge(treeptr.nextptr[nchild]);
		}

		return bhasmerge;
	}

	// ////////////////////////////////////////////////////////////////////
	/**
	 * Returns a string with the model parameters
	 */
	public String saveString(Treenode treecopy) {
		StringBuffer szbuf = new StringBuffer("");
		;

		bhasmerge = hasMerge(treecopy);

		if (bhasmerge) {
			szbuf.append("MERGED TREE\n");
		}

		if (BREGDREM) {
			szbuf.append("REGDREM Num. Coefficients\t"
					+ bindingData.regNames.length + "\n");
		} else {
			szbuf.append("Num. Coefficients\t" + bindingData.regNames.length
					+ "\n");
		}

		HashMap<Treenode, Integer> ptrMap = new HashMap<Treenode, Integer>();
		ntotalID = -1;
		szbuf.append(saveStringHelp(treecopy, ptrMap));
		return szbuf.toString();
	}

	/**
	 * Helper function for generating a string with model parameters that
	 * traverses all the states
	 */
	private StringBuffer saveStringHelp(Treenode ptr,
			HashMap<Treenode, Integer> ptrMap) {
		StringBuffer szbuf = new StringBuffer();
		if (ptr != null) {
			if (bhasmerge) {
				Integer obj = ptrMap.get(ptr);
				int nID;
				if (obj != null) {
					nID = obj.intValue();
				} else {
					ntotalID++;
					nID = ntotalID;
					ptrMap.put(ptr, new Integer(nID));
				}
				szbuf.append(nID + "\t");
			}

			szbuf.append(ptr.dmean + "\t" + ptr.dsigma + "\t" + ptr.numchildren
					+ "\n");
			if (ptr.numchildren > 1) {
				if (BREGDREM) {
					for (int nchild = 0; nchild < ptr.numchildren; nchild++) {
						szbuf.append(ptr.ptrans[nchild] + "\n");
					}
				} else {
					szbuf
							.append(ptr.tranC
									.saveClassifier(bindingData.regNames));
				}

			}

			for (int nindex = 0; nindex < ptr.numchildren; nindex++) {
				if (ptr.nextptr[nindex] != null) {
					szbuf.append(saveStringHelp(ptr.nextptr[nindex], ptrMap));
				}
			}
		}

		return szbuf;
	}

	/**
	 * Save the activity scores at this node and its children
	 * 
	 * @return
	 */
	public StringBuffer saveActivityScores(Treenode node) {
		StringBuffer buf = new StringBuffer();

		if (node != null && node.numchildren >= 1) {
			if (node.numchildren >= 2) {
				for (int t = 0; t < bindingData.regNames.length; t++) {
					buf.append(bindingData.regNames).append("\t");
					buf.append(node.dTFActivityScore[t]).append("\t");
					buf.append(node.ndepth).append("\n");
				}
			}

			for (int nindex = 0; nindex < node.numchildren; nindex++) {
				if (node.nextptr[nindex] != null) {
					buf.append(saveActivityScores(node.nextptr[nindex]));
				}
			}
		}

		return buf;
	}

	// ////////////////////////////////////////////////////////////////////
	/**
	 * Record of information about predictions
	 */
	static class PredictRec {
		double[] mean;
		double[] meansq;
		double[] var;
		double[] entropy;
	}

	/**
	 * Used to set probability of each state a gene with a TF-binding signature
	 * of binding will be in Also computes a record of other statistics based on
	 * these predictions
	 */
	public PredictRec predictTime(int[] binding, double dclassprob,
			DREM_Timeiohmm.Treenode treeptr) throws Exception {
		PredictRec thePredictRec = new PredictRec();
		thePredictRec.mean = new double[traindata[0].length];
		thePredictRec.var = new double[traindata[0].length];
		thePredictRec.meansq = new double[traindata[0].length];
		thePredictRec.entropy = new double[traindata[0].length];
		predictTimeHelp(treeptr, binding, dclassprob, 0, thePredictRec.mean,
				thePredictRec.var, thePredictRec.meansq, thePredictRec.entropy);

		return thePredictRec;

	}

	/**
	 * A recursive helper function to predictTime
	 */
	private void predictTimeHelp(Treenode node, int[] theInstance,
			double dweight, int nstep, double[] meanpredict,
			double[] varpredict, double[] meansqpredict, double[] entropypredict)
			throws Exception {
		if (nstep < meanpredict.length) {
			NumberFormat nf1 = NumberFormat.getInstance(Locale.ENGLISH);

			if (dweight < 1) {
				nf1.setMaximumIntegerDigits(0);
				nf1.setMinimumFractionDigits(2);
				nf1.setMaximumFractionDigits(2);
			} else {
				nf1.setMinimumFractionDigits(1);
				nf1.setMaximumFractionDigits(1);
			}
			node.dpredictweight = dweight;

			if (node.thepredictText != null) {
				node.thepredictText.setText(nf1.format(dweight));
			}

			meanpredict[nstep] += dweight * node.dmean;
			varpredict[nstep] += Math.pow(dweight * node.dsigma, 2);
			meansqpredict[nstep] += dweight * node.dmean * node.dmean;
			entropypredict[nstep] -= dweight * Math.log(dweight) / Math.log(2);

			if (node.binit) {
				double[] pdist;
				if (BREGDREM) {
					pdist = node.ptrans;
				} else {
					pdist = node.tranC.distributionForInstance(theInstance);
				}

				if (BDEBUG) {
					if (pdist.length == 1)
						System.out.println("warning " + pdist[0]);
				}

				for (int nchild = 0; nchild < node.numchildren; nchild++) {
					predictTimeHelp(node.nextptr[nchild], theInstance, dweight
							* pdist[nchild], nstep + 1, meanpredict,
							varpredict, meansqpredict, entropypredict);
				}
			} else {
				predictTimeHelp(node.nextptr[0], theInstance, dweight,
						nstep + 1, meanpredict, varpredict, meansqpredict,
						entropypredict);
			}
		}
	}

	// /////////////////////////////////////////////////////////////////////
	/**
	 * Used for debugging to see the contents of the model
	 */
	public void traverse(Treenode root, int ndepth, boolean bprintC) {
		if (BDEBUG) {
			for (int nindex = 0; nindex < ndepth; nindex++) {
				System.out.print("\t");
			}

			System.out.print(root.numPath + "\t" + root.dmean + "\t"
					+ root.dsigma + "\t" + root.numchildren + "\t" + root.binit
					+ "\t" + root.dpredictweight + "\t" + root.ndepth + "\t#"
					+ root.numdownleaves + "\t");

			if (root.parent != null)
				System.out.print("p " + root.parent.dmean);

			if (root.parentptrA != null) {
				for (int i = 0; i < root.parentptrA.length; i++)
					System.out.print("\t"
							+ nf2.format(root.parentptrA[i].dmean));
			}

			if (root.ptrans != null)
				System.out.print("\t" + root.ptrans[0] + "\t" + root.ptrans[1]);

			for (int nindex = ndepth; nindex < theDataSet.data[0].length - 1; nindex++) {
				System.out.print("\t");
			}

			if (root.nextptr[0] != null) {
				System.out.println();

				if ((bprintC) && (root.numchildren > 1)) {
					System.out.println(root.tranC);
				}
			} else {
				System.out.println();
			}

			for (int nindex = 0; nindex < root.numchildren; nindex++) {
				if (root.nextptr[nindex] != null) {
					traverse(root.nextptr[nindex], ndepth + 1, bprintC);
				}
			}
		}
	}

	// ///////////////////////////////////////////////////////////////////////

	/**
	 * Combines the results of the forward and backward algorithm to obtain the
	 * probability of each gene going through each state when using viterbi
	 * training
	 */
	public void instanceAEV(Treenode root, double[] vals, int[] pma,
			double[] bpvals, int[] bestpath, double dpj, int nrec)
			throws Exception {
		Treenode tempptr = root;

		for (int ndepth = 0; ndepth < bestpath.length; ndepth++) {
			tempptr.dEsum += vals[ndepth];
			tempptr.dPsum += 1;
			tempptr.dEsumsq += Math.pow(vals[ndepth], 2);
			tempptr.nprime++;

			if (BREGDREM) {
				tempptr.aprobs[bestpath[ndepth]]++;
			}

			if (tempptr.numchildren >= 1) {
				tempptr.recweight[tempptr.numchildren * nrec + bestpath[ndepth]] = 1;
			}

			tempptr = tempptr.nextptr[bestpath[ndepth]];
		}
	}

	// /////////////////////////////////////////////////////////////////////////////////
	/**
	 * Initializes all the fields that will be used by instanceAE
	 */
	public void initAE(Treenode root) {
		if (root != null) {
			for (int nchild = 0; nchild < root.numchildren; nchild++) {
				if (BREGDREM) {
					root.aprobs[nchild] = 0;
				}
				initAE(root.nextptr[nchild]);
			}

			if ((root.recweight == null)
					|| (root.recweight.length != root.numchildren
							* traindata.length)) {
				root.recweight = new double[root.numchildren * traindata.length];
			}

			root.dEsum = 0;
			root.dEsumsq = 0;
			root.dPsum = 0;
			root.nprime = 0;
			root.niteration = 0;
			root.ninstancerec = 0;
			root.ninstanceiteration = 0;
			root.nrec = -1;
		}
	}

	/**
	 * Combines the results of the forward and backward algorithm to obtain the
	 * probability of each gene going through each state when using baum-welch
	 * training
	 */
	public void instanceAE(Treenode root, int ndepth, double[] vals, int[] pma,
			double dpj, int nrec) throws Exception {
            
		if ((root.ninstancerec != nrec)
				|| (root.ninstanceiteration != nglobaliteration)) {
			root.ninstanceiteration = nglobaliteration;
			root.ninstancerec = nrec;

			if (root.nextptr[0] != null) {
				for (int nchild = 0; nchild < root.numchildren; nchild++) {
					double dmultval = root.nextptr[nchild].df
							* root.nextptr[nchild].db / dpj;

					// P(A,B)P(C|B)=P(A,B)P(C|A,B)=P(A,B,C)

					// double df; //p(x_1,...,x_i,pi_i=k)
					// double db; //p(x_(i+1),...,x_L|pi_i=k)
					// p(x_1,...,x_L,pi_i=k)
					// =p(x_1,...,x_i,pi_i=k)*p(x_(i+1),...,x_L|x_1,...,x_i,pi_i=k)
					// =p(x_1,...,x_i,pi_i=k)*p(x_(i+1),...,x_L|pi_i=k)
					// p(x_1,...,x_L) = p(pi_i=k|x_1,...x_L)
                                        /*
                                        //FIXME NANTEST
                                        if (Double.isInfinite(root.nextptr[nchild].db)){
                                            dmultval = root.nextptr[nchild].df*1.0/1.0;
                                        }
                                        **/
                                        
					root.recweight[root.numchildren * nrec + nchild] = dmultval;

					if (BREGDREM) {
						root.aprobs[nchild] += dmultval;
					}
                                        
					instanceAE(root.nextptr[nchild], ndepth + 1, vals, pma,
							dpj, nrec);
				}
			}

			double dweight = root.df * root.db / dpj;
			if ((dweight > 0) && (pma[ndepth] != 0)
					&& (!Double.isInfinite(dweight))) {

				root.dEsum += vals[ndepth] * dweight;
				root.dPsum += dweight;
				root.dEsumsq += vals[ndepth] * vals[ndepth] * dweight;
				root.nprime++;
			}
		}
	}

	// ///////////////////////////////////////////////////////////////////////
	/**
	 * Initializes all the fields that will be used by instanceAEV
	 */
	public void initAEV(Treenode root) {
		if (root != null) {
			if (BREGDREM) {
				for (int nchild = 0; nchild < root.aprobs.length; nchild++) {
					root.aprobs[nchild] = 0;
				}
			}

			for (int nchild = 0; nchild < root.numchildren; nchild++) {
				if (BREGDREM) {
					root.aprobs[nchild] = 0;
				}
				initAEV(root.nextptr[nchild]);
			}

			if ((root.recweight == null)
					|| (root.recweight.length != root.numchildren
							* traindata.length)) {
				root.recweight = new double[root.numchildren * traindata.length];
			} else {
				for (int i = 0; i < root.recweight.length; i++) {
					root.recweight[i] = 0;
				}
			}

			root.dEsum = 0;
			root.dEsumsq = 0;
			root.dPsum = 0;
			root.nprime = 0;
		}
	}

	// //////////////////////////////////////////////////////////////////////
	/**
	 * Forces children of a parent to both have the average of their standard
	 * deviations
	 */
	public void averageChildrenSigmas(Treenode root) {
		if (root.numchildren >= 2) {
			double dsigmasum = 0;
			double dpsum = 0;
			for (int nchild = 0; nchild < root.numchildren; nchild++) {
				dsigmasum += root.nextptr[nchild].dPsum
						* root.nextptr[nchild].dsigma;
				dpsum += root.nextptr[nchild].dPsum;
				if (BDEBUG) {
					System.out.println("##" + root.nextptr[nchild].dsigma
							+ "\t" + root.nextptr[nchild].dPsum);
				}
			}
			double dsigmaavg = dsigmasum / dpsum;
			if (BDEBUG) {
				System.out.println("&&&" + dpsum + "\t" + root.dPsum + "\t"
						+ dsigmasum + "\t" + dsigmaavg);
			}

			for (int nchild = 0; nchild < root.numchildren; nchild++) {
				root.nextptr[nchild].dsigma = dsigmaavg;
			}
		}

		for (int nchild = 0; nchild < root.numchildren; nchild++) {
			averageChildrenSigmas(root.nextptr[nchild]);
		}
	}

	// ///////////////////////////////////////////////////////////////////////

	/**
	 * Updates the mean, standard deviation, and classifier parameters based on
	 * the current expectation of which genes will go through which paths
	 */
	public void updateParams(Treenode root, int nchildnum, double[][] data,
			int[][] pma, int ndepth) throws Exception {
		if (!root.bchange) {
			// not allowed to change states params
			// maybe we can get rid of this below
			// since if there is no change in root can assume all children not
			// changeable
			if (root.nextptr[0] != null) {
				for (int nchild = 0; nchild < root.numchildren; nchild++) {
					updateParams(root.nextptr[nchild], nchild, data, pma,
							ndepth + 1);
				}
			}
		} else {

			double doldmean = root.dmean;

			if (root.dPsum > 0) {
				root.dmean = root.dEsum / root.dPsum;
			} else {
				root.dmean = 0;
			}

			if (Double.isNaN(root.dmean)) {
				System.out.println("not a valid mean " + root.dEsum + "\t"
						+ root.dPsum + "\t" + doldmean);
				System.out.println(root.tranC);
				System.out.println("--------------------");
				throw new IllegalArgumentException("not a valid mean");
			}
			double tempdval = -10;
			if (root.nprime <= 1) {
				root.dsigma = Math.max(dminstddev, DEFAULTSIGMA);
			} else {
				double dval = (double) root.nprime
						/ (double) (root.nprime - 1)
						* (root.dEsumsq / root.dPsum - Math.pow(root.dEsum
								/ root.dPsum, 2));
				if (dval <= 0) {
					root.dsigma = Math.max(dminstddev, DEFAULTSIGMA);
				} else {
					// fix here to handle floating point error if standard
					// deviation was too low
					root.dsigma = Math.max(dminstddev, Math.sqrt(dval));
					// root.dsigma = Math.sqrt(dval);
				}
				tempdval = dval;
			}

			if (!(root.dsigma > 0)) {
				System.out.println("## " + root.dsigma + " " + root.nprime
						+ " " + root.dEsumsq + " " + root.dEsum + " "
						+ root.dPsum + " " + dminstddev + " " + tempdval + " "
						+ DEFAULTSIGMA);
			}

			if (root.nextptr[0] != null) {
				if (root.numchildren > 1) {
					if (BREGDREM) {
						root.binit = true;
					} else {
						if (root.tranC == null) {
							root.tranC = new DREM_FastLogistic2(
									trainBinding.theInstancesIndex[root.ndepth][root.numchildren - 2],
									trainBinding.theInstances[root.ndepth][root.numchildren - 2],
									trainBinding.theInstancesRegIndex[root.ndepth][root.numchildren - 2],
									trainBinding.theInstancesReg[root.ndepth][root.numchildren - 2],
									trainBinding.ylabels[root.ndepth][root.numchildren - 2],
									root.recweight, root.numchildren, numbits,
									trainBinding.regTypes);
							root.tranC.setRidge(RIDGE);
						} else {
							root.tranC
									.reinit(
											trainBinding.theInstancesIndex[root.ndepth][root.numchildren - 2],
											trainBinding.theInstances[root.ndepth][root.numchildren - 2],
											trainBinding.theInstancesRegIndex[root.ndepth][root.numchildren - 2],
											trainBinding.theInstancesReg[root.ndepth][root.numchildren - 2],
											trainBinding.ylabels[root.ndepth][root.numchildren - 2],
											root.recweight, root.numchildren,
											trainBinding.regTypes);
						}

						/*
						 * before doing this set the param in the classifier
						 * that tells which next node is the high and which the
						 * low node
						 * 
						 * CASE -1 0smaller1: ptr.nextptr[0].dmean <
						 * ptr.nextptr[1].dmeanchild -> coeff < 0 CASE +1
						 * 1smaller0: ptr.nextptr[0].dmean >
						 * ptr.nextptr[1].dmeanchild -> coeff > 0 CASE +2
						 * bothUp: if miRNA downregulated -> coeff [-inf,+inf]
						 * CASE -2 bothDown: if miRNA upregulated -> coeff
						 * [-inf,+inf]
						 */
						/*
						 * seems to lead to underfitting because too much
						 * influence on the model if(root.nextptr[0].dmean >
						 * root.dmean && root.nextptr[0].dmean > root.dmean){
						 * root.tranC.childExpStatus = 2; } else
						 * if(root.nextptr[0].dmean < root.dmean &&
						 * root.nextptr[0].dmean < root.dmean){
						 * root.tranC.childExpStatus = -2; } else
						 */
						if (root.nextptr[0].dmean > root.nextptr[1].dmean) {
							root.tranC.childExpStatus = 1;
						} else {
							root.tranC.childExpStatus = -1;
						}
						
						root.tranC.train();

						root.binit = true;
					}
				}

				double dsum = 0;

				if (BREGDREM) {
					for (int nchild = 0; nchild < root.numchildren; nchild++) {
						if (root.nextptr[nchild].bchange) {
							dsum += root.aprobs[nchild];
						}
					}
				}

				for (int nchild = 0; nchild < root.numchildren; nchild++) {
					if (BREGDREM) {
						if (dsum > 0) {
							root.ptrans[nchild] = root.aprobs[nchild] / dsum;
							if (BDEBUG) {
								System.out.println("probs: "
										+ root.aprobs[nchild] + " " + dsum
										+ " " + root.ptrans[nchild]);
							}
						} else {
							root.ptrans[nchild] = 0;
						}

						if (Double.isNaN(root.ptrans[nchild]))
							System.out.println("%%%%%% " + root.ptrans[nchild]
									+ "\t" + root.aprobs[nchild] + "\t" + dsum);
					}
					updateParams(root.nextptr[nchild], nchild, data, pma,
							ndepth + 1);
				}
			}
		}
	}

	// ////////////////////////////////////////////////////////////////////////////
	/**
	 * Executes the forward algorithm portion of the baum-welch algorithm
	 */
	public void forwardalg(double[] vals, int[] pma,
			int[][][] theInstanceIndex, double[][][] theInstance, int ni,
			double da, Treenode node, int nrec, Treenode parentptr,
			int instanceindex) throws Exception {
		// da is trans prob
		// f_k(i) = P(x_1,...,x_i,pi_i=k)
		// f_0(0) = 1, f_k(0)=0 for k>0

		if (node != null) {
			// sum_m P(x_1,...,x_(i-1)|pi_(i-1)=m)a_(mk)*prob(vals[ni])
			// but were only assuming there is one path to each state

			if (pma[ni] != 0) {
				double dxmudiff = (vals[ni] - node.dmean);
				node.dens = Math.exp(-dxmudiff * dxmudiff
						/ (2 * node.dsigma * node.dsigma))
						/ (node.dsigma * StatUtil.TWOPISQRT);
			} else {
				node.dens = 1;
			}

			if (node.parentptrA == null) {
				node.df = node.parent.df * da * node.dens;
			} else {
				if ((node.nrec != nrec)
						|| (node.niteration != nglobaliteration)) {
					node.df = parentptr.df * da * node.dens;
					if (node.niteration != nglobaliteration)
						node.niteration = nglobaliteration;
					if (node.nrec != nrec)
						node.nrec = nrec;
				} else {
					node.df += parentptr.df * da * node.dens;
					node.niteration++;
				}
			}

			double[] ptranlogit;

			if (BREGDREM) {
				ptranlogit = node.ptrans;
			} else if ((node.numchildren > 1) && (node.binit)) {
				ptranlogit = node.tranC.distributionForInstance(
						theInstanceIndex[node.ndepth][instanceindex],
						theInstance[node.ndepth][instanceindex]);
			} else {
				ptranlogit = CONSTANTA[node.numchildren];
			}

			for (int nchild = 0; nchild < node.numchildren; nchild++) {
				try {
					forwardalg(vals, pma, theInstanceIndex, theInstance,
							ni + 1, ptranlogit[nchild], node.nextptr[nchild],
							nrec, node, instanceindex);
				} catch (Exception ex) {
					ex.printStackTrace();
					System.out.println(ptranlogit.length + " ------abc--"
							+ node.nextptr[0] + "------------- "
							+ node.nextptr.length + " " + nchild + "\t**"
							+ node.numchildren + "\t" + node.tranC.numclasses
							+ "\t" + node.binit + "\t" + node.dmean + "\t"
							+ node.dsigma);
					System.out.println(node.tranC);
				}
			}
		}
	}

	// ///////////////////////////////////////////////////////////////////////////
	/**
	 * Executes the backward portion of the baum-welch algorithm
	 */
	public void backalg(double[] vals, int[] pma, int[][][] theInstanceIndex,
			double[][][] theInstance, int ni, Treenode node, int nrec,
			boolean bforward, int instanceindex) throws Exception {
		// b_k(i) = P(x_(i+1)...x_L|pi=k)
		// P(x_1,...,x_i,pi_i=k)
		if (node.nextptr[0] == null) {
			node.db = 1;
		} else {
			node.db = 0;
			double[] ptranlogit;

			if (BREGDREM) {
				ptranlogit = node.ptrans;
			} else if ((node.numchildren > 1) && (node.binit)) {
				ptranlogit = node.tranC.distributionForInstance(
						theInstanceIndex[node.ndepth][instanceindex],
						theInstance[node.ndepth][instanceindex]);
			} else {
				ptranlogit = CONSTANTA[node.numchildren];
			}

			for (int nchild = 0; nchild < node.numchildren; nchild++) {
				Treenode nextnode = node.nextptr[nchild];
				backalg(vals, pma, theInstanceIndex, theInstance, ni + 1,
						nextnode, nrec, bforward, instanceindex);

				double dens;

				if (bforward) {
					dens = nextnode.dens;
				} else if (pma[ni + 1] != 0) {
					double dxmudiff = (vals[ni + 1] - nextnode.dmean);
					dens = Math.exp(-dxmudiff * dxmudiff
							/ (2 * nextnode.dsigma * nextnode.dsigma))
							/ (nextnode.dsigma * StatUtil.TWOPISQRT);
				} else {
					dens = 1;
				}

				if (BDEBUG) {
					if (nchild >= ptranlogit.length) {
						System.out.println("**********" + nchild + "\t"
								+ ptranlogit.length);
					}
				}

				node.db += ptranlogit[nchild] * nextnode.db * dens;
                                
                                /**
                                //FIXME: NANTEST   
                                if (Double.isInfinite(node.db)){
                                                               throw new Exception();
                                                               //System.out.println(dpj);
                                                           }
                                **/
                                
				if (node.db < 0) {
					System.out.println("##" + node.db + "\t" + nextnode.dens);
					for (int i = 0; i < ptranlogit.length; i++)
						System.out.print(ptranlogit[i] + "\t");
					System.out.println();

				}
			}
		}
	}

	// //////////////////////////////////////////////////////////////
	/**
	 * Clears any prior assignments of genes to paths through the model
	 */
	public void clearCounts(Treenode ptr) {
		if (ptr != null) {
			ptr.bInNode = new boolean[theDataSet.data.length];
			ptr.numPath = 0;

			if (bindingData.gene2RegBinding != null) {
				ptr.htScore = (Hashtable<Double, Integer>[]) new Hashtable[numbits];
				ptr.dTotalScore = new double[numbits];

				for (int nbindingindex = 0; nbindingindex < ptr.htScore.length; nbindingindex++) {
					ptr.htScore[nbindingindex] = new Hashtable<Double, Integer>();
				}
			}
			for (int nchild = 0; nchild < ptr.numchildren; nchild++) {
				clearCounts(ptr.nextptr[nchild]);
			}
		}
	}

	// ////////////////////////////////////////////////////////////////////////////
	/**
	 * Determines the most likely path of each gene through the model
	 */
	public void viterbi(Treenode treeptr) throws Exception {
		int[] bestpath;
		bestpath = new int[theDataSet.data[0].length];
		storedbestpath = new BigInteger[theDataSet.data.length];
		clearCounts(treeptr);
		treeptr.numPath = theDataSet.data.length;
		if (BDEBUG) {
			System.out.print("!!!!!!" + theDataSet.data.length);
			if (bindingData.gene2RegBinding != null)
				System.out.println(" "
						+ bindingData.gene2RegBinding[treeptr.ndepth].length);
		}

		for (int nrow = 0; nrow < theDataSet.data.length; nrow++) {
			if (bindingData.gene2RegBinding != null) {
				computevlogit(theDataSet.data[nrow],
						theDataSet.pmavalues[nrow],
						bindingData.gene2RegBinding,
						bindingData.gene2RegBindingIndex, 0, treeptr, bestpath,
						nrow);
			} else {
				computevlogit(theDataSet.data[nrow],
						theDataSet.pmavalues[nrow], null, null, 0, treeptr,
						bestpath, 0);
			}

			BigInteger nsum = new BigInteger("0");
			Treenode tempptr = treeptr;

			tempptr.bInNode[nrow] = true;

			for (int nindex = 0; nindex < bestpath.length - 1; nindex++) {
				int currdepth = tempptr.ndepth;
				tempptr = tempptr.nextptr[bestpath[nindex]];
				tempptr.bInNode[nrow] = true;
				tempptr.numPath++;

				if (bindingData.gene2RegBinding != null) {
					int ntfindex = 0;
					for (int nbit = 0; nbit < numbits; nbit++) {
						while ((ntfindex < bindingData.gene2RegBindingIndex[currdepth][nrow].length)
								&& (nbit > bindingData.gene2RegBindingIndex[currdepth][nrow][ntfindex])) {
							ntfindex++;
						}

						Double objMap;
						if ((ntfindex < bindingData.gene2RegBindingIndex[currdepth][nrow].length)
								&& (nbit == bindingData.gene2RegBindingIndex[currdepth][nrow][ntfindex])) {
							tempptr.dTotalScore[nbit] += bindingData.gene2RegBinding[currdepth][nrow][ntfindex];
							objMap = new Double(
									bindingData.gene2RegBindingSigned[currdepth][nrow][ntfindex]);
						} else {
							objMap = new Double(0.0);
						}
						Integer objCount = tempptr.htScore[nbit].get(objMap);
						if (objCount != null) {
							tempptr.htScore[nbit].put(objMap, new Integer(
									objCount.intValue() + 1));
						} else {
							tempptr.htScore[nbit].put(objMap, new Integer(1));
						}
					}
				}
				// szsum += ""+bestpath[nindex];
				// nsum += (int)
				// Math.pow(nmaxchild,bestpath.length-nindex-1)*bestpath[nindex];
				nsum = nsum.add((new BigInteger("" + nmaxchild)).pow(
						bestpath.length - nindex - 1).multiply(
						new BigInteger("" + bestpath[nindex])));
			}
			storedbestpath[nrow] = nsum;
		}
	}

	// ////////////////////////////////////////////////////////////////////////////////
	/**
	 * Computes association scores for the transcription factors overall on
	 * paths and on paths out of splits conditional on the set of genes going
	 * into the split
	 */
	public void computeStats(Treenode treeptr, Treenode rootptr) {
		if (treeptr != null) {
			treeptr.dscore = new double[numbits][treeptr.numchildren];
			treeptr.ddiff = new double[numbits][treeptr.numchildren];
			treeptr.dpvals = new double[numbits][treeptr.numchildren];

			treeptr.ncountvals = new int[numbits][treeptr.numchildren][bindingData.signedBindingValuesSorted.length];

			if (treeptr.numchildren >= 3) {
				treeptr.nothersum = new int[numbits][treeptr.numchildren][bindingData.signedBindingValuesSorted.length];
				treeptr.dscoreother = new double[numbits][treeptr.numchildren];
			}

			int[][] ncountgrid = new int[2][bindingData.signedBindingValuesSorted.length];
			treeptr.ncountTotals = new int[numbits][bindingData.signedBindingValuesSorted.length];

			treeptr.dTFActivityScore = new double[numbits];

			treeptr.tsSigTFActivity = (TreeSet<SigTFRecv2>[]) new TreeSet[treeptr.numchildren];
			for(int i = 0; i < treeptr.numchildren; i++)
				treeptr.tsSigTFActivity[i] = new TreeSet<SigTFRecv2>(
						new SigTFRecv2Compare());
			
			treeptr.tsSigTF = new TreeSet<SigTFRecv2>(new SigTFRecv2Compare());
			for (int ntf = 0; ntf < numbits; ntf++) {
				double dnumpaths = 0;
				double dscoretotal = 0;
				for (int nindex = 0; nindex < treeptr.ncountTotals[ntf].length; nindex++) {
					treeptr.ncountTotals[ntf][nindex] = 0;
				}

				for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
					if (treeptr.nextptr[nchild].htScore == null) {
						for (int nvalindex = 0; nvalindex < treeptr.ncountvals[ntf][nchild].length; nvalindex++) {
							treeptr.ncountvals[ntf][nchild][nvalindex] = 0;
						}
					} else {
						for (int nel = 0; nel < bindingData.signedBindingValuesSorted.length; nel++) {
							Double key = new Double(
									(double) bindingData.signedBindingValuesSorted[nel]);
							Integer objcount = treeptr.nextptr[nchild].htScore[ntf]
									.get(key);

							if (objcount != null) {
								int ntempval = ((Integer) objcount).intValue();
								treeptr.ncountvals[ntf][nchild][nel] = ntempval;
								treeptr.ncountTotals[ntf][nel] += ntempval;
							} else {
								treeptr.ncountvals[ntf][nchild][nel] = 0;
							}
						}
					}

					treeptr.dscore[ntf][nchild] = treeptr.nextptr[nchild].dTotalScore[ntf]
							/ treeptr.nextptr[nchild].numPath;
					dnumpaths += treeptr.nextptr[nchild].numPath;
					dscoretotal += treeptr.nextptr[nchild].dTotalScore[ntf];
				}

				if (treeptr.numchildren == 2) {
					treeptr.ddiff[ntf][0] = treeptr.dscore[ntf][0]
							- treeptr.dscore[ntf][1];
					treeptr.dpvals[ntf][0] = computepval(
							treeptr.ncountvals[ntf], Math
									.abs(treeptr.ddiff[ntf][0]));
					treeptr.tsSigTF.add(new SigTFRecv2(
							bindingData.regNames[ntf], treeptr.dpvals[ntf][0]));

				} else if (treeptr.numchildren >= 3) {
					for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
						// divide by 0 need to check about that
						double ddiff = (dnumpaths - treeptr.nextptr[nchild].numPath);
						double davgother;

						if (ddiff > 0) {
							davgother = (dscoretotal - treeptr.nextptr[nchild].dTotalScore[ntf])
									/ ddiff;
						} else {
							davgother = 0;
						}
						treeptr.dscoreother[ntf][nchild] = davgother;

						treeptr.ddiff[ntf][nchild] = treeptr.dscore[ntf][nchild]
								- davgother;

						for (int nindex = 0; nindex < treeptr.ncountTotals[ntf].length; nindex++) {
							ncountgrid[0][nindex] = treeptr.ncountvals[ntf][nchild][nindex];
							ncountgrid[1][nindex] = treeptr.ncountTotals[ntf][nindex]
									- ncountgrid[0][nindex];
							treeptr.nothersum[ntf][nchild][nindex] = ncountgrid[1][nindex];
						}

						treeptr.dpvals[ntf][nchild] = computepval(ncountgrid,
								Math.abs(treeptr.ddiff[ntf][nchild]));
					}

					for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
						treeptr.tsSigTF.add(new SigTFRecv2(
								bindingData.regNames[ntf],
								treeptr.dpvals[ntf][treeptr.orderA[nchild]]));
					}
				}
				// Activity scores are only calculated if there is a split at
				// this node
				if (treeptr.numchildren > 1 && bindingData.regPriors != null) {
					/**
					 * The number of genes on a particular path out of a split
					 * regulated by a TF. Regulated means the interaction value
					 * is not 0. Dimension:<br>
					 * [num paths out of split]
					 */
					int[] nRegThisPath = new int[treeptr.numchildren];
					/**
					 * The number of genes on all paths except the given path
					 * out of a split that are regulated by a TF. Regulated
					 * means the interaction value is not 0. Dimension:<br>
					 * [num paths out of split]
					 */
					int[] nRegOtherPaths = new int[treeptr.numchildren];
					/**
					 * The number of genes on all paths out of a split that are
					 * regulated by a TF. Regulated means the interaction value
					 * is not 0.
					 */
					int nRegAllPaths = 0;

					// Initialize nRegThisPath
					for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
						nRegThisPath[nchild] = 0;

					}

					// Populate nRegThisPath and nRegAllPaths
					for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
						for (int nel = 0; nel < bindingData.signedBindingValuesSorted.length; nel++) {
							// Only count genes the TF regulates, which are
							// those
							// genes with an interaction value != 0
							if (bindingData.signedBindingValuesSorted[nel] != 0) {
								nRegThisPath[nchild] += treeptr.ncountvals[ntf][nchild][nel];
								nRegAllPaths += treeptr.ncountvals[ntf][nchild][nel];
							}
						}
					}

					// Populate nRegOtherPaths
					for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
						nRegOtherPaths[nchild] = nRegAllPaths
								- nRegThisPath[nchild];
					}

					// Now determine which path is the primary path out of the
					// split.
					// The primary path is the path with the most regulated
					// genes. In the case of the tie, the path with the fewest
					// genes
					// overall out of the split is the primary path.
					int nMaxPathVal = -1;
					for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
						nMaxPathVal = Math.max(nMaxPathVal,
								nRegThisPath[nchild]);
					}

					// See how many paths have this max value
					boolean[] bMaxPaths = new boolean[treeptr.numchildren];
					int nMaxPathCount = 0;
					for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
						if (nRegThisPath[nchild] == nMaxPathVal) {
							bMaxPaths[nchild] = true;
							nMaxPathCount++;
						} else {
							bMaxPaths[nchild] = false;
						}
					}

					// The index of the primay path
					int nPrimPathInd = -1;
					if (nMaxPathCount == 1) {
						for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
							if (bMaxPaths[nchild]) {
								nPrimPathInd = nchild;
							}
						}
					} else {
						// There was a tie so look at all paths out of the split
						// there were involved in the tie
						// to find the path with the fewest genes
						// If there is another tie, it won't affect the score
						// calculation so pick the first one found
						int nMinPathVal = Integer.MAX_VALUE;

						for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
							if (bMaxPaths[nchild]
									&& treeptr.nextptr[nchild].numPath < nMinPathVal) {
								nMinPathVal = treeptr.nextptr[nchild].numPath;
								nPrimPathInd = nchild;
							}
						}
					}

					// Calculate the number of genes on the paths that are not
					// the primary path
					int nNumOtherPaths = 0;
					for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
						if (nPrimPathInd != nchild) {
							nNumOtherPaths += treeptr.nextptr[nchild].numPath;
						}
					}

					// The formula for a TF's activity score is:
					// score = numer / denom
					// where numer = (prob binding is functional)^(num genes TF
					// regulates on primary path)
					// * (prob binding is not functional)^(num genes TF
					// regulates on other paths)
					// * (TF activity prior)
					// and denom = (ratio of all genes on TFs primary path)^(num
					// genes TF regulates on primary path)
					// * (ratio of all genes on other paths)^(num genes TF
					// regulates on other paths)
					// * (1 - TF activity prior)
					// Calculations need to be done in logspace, otherwise if there are too
					// many bound genes the score can go to 0 or NaN incorrectly
					double scoreNumer = Math.log(dProbBindingFunctional) *
							nRegThisPath[nPrimPathInd];
					scoreNumer += Math.log(1 - dProbBindingFunctional) *
							nRegOtherPaths[nPrimPathInd];
					// TODO: multiply activity prior by magnitude of mirna score???
					scoreNumer += Math.log(bindingData.regPriors[ntf]);
					double scoreDenom = Math.log(
							treeptr.nextptr[nPrimPathInd].numPath
									/ ((double) treeptr.numPath)) *
							nRegThisPath[nPrimPathInd];
					scoreDenom += Math.log(nNumOtherPaths
							/ ((double) treeptr.numPath)) *
							nRegOtherPaths[nPrimPathInd];
					scoreDenom += Math.log(1 - bindingData.regPriors[ntf]);
					// Take out of logspace for display and output
					treeptr.dTFActivityScore[ntf] = Math.exp(scoreNumer - scoreDenom);
					
					// Max activity score is either the new TF activity score at
					// this node
					// or the pre-existing max activity score
					dMaxTFActivityScore[ntf] = Math.max(
							treeptr.dTFActivityScore[ntf],
							dMaxTFActivityScore[ntf]);

					// Add the activity score at this node to the TreeSet
					// Instead of storing the score directly, store it in a form
					// that resembles
					// a pvalue so that the existing signifance score threshold
					// and sorting
					// code work
					double pseudoPvalue = Double.POSITIVE_INFINITY;
					if (treeptr.dTFActivityScore[ntf] != 0) {
						pseudoPvalue = 1 / treeptr.dTFActivityScore[ntf];
					}
					// Also with the TF name, store the index of the primary
					// path for that
					// TF coming out of the split
					// TODO store the path index separately
					treeptr.tsSigTFActivity[nPrimPathInd]
							.add(new SigTFRecv2(bindingData.regNames[ntf], pseudoPvalue));
				} else // treeptr.numchilren is 0 or 1 so there is no split to
				// calculate the score
				{
					treeptr.dTFActivityScore[ntf] = 0;
					// No need to add a SigTFRecv2 object to the tsSigTFActivity
					// TreeSet
					// if we are not at a split
				}

			} // end loop through all TFs

			int numrowstable;
			int nsize = bindingData.existingBindingValuesUnsorted.size() - 1;
			//numrowstable = bindingData.regNames.length * nsize;
                        numrowstable=bindingData.regNames.length;
			boolean bsplit;
			if (treeptr.numchildren >= 2) {
				treeptr.dpvalEdgeSplit = new double[treeptr.numchildren][numrowstable];
				treeptr.ddiffEdgeSplit = new double[treeptr.numchildren][numrowstable];
				treeptr.dexpectEdgeSplit = new double[treeptr.numchildren][numrowstable];
				treeptr.tsSigTFEdgeSplit = (TreeSet<SigTFRecv2>[]) new TreeSet[treeptr.numchildren];
				treeptr.dratioSplit = new double[treeptr.numchildren][numrowstable];

				for (int nindex = 0; nindex < treeptr.numchildren; nindex++) {
					treeptr.tsSigTFEdgeSplit[nindex] = new TreeSet<SigTFRecv2>(
							new SigTFRecv2Compare());
				}
				bsplit = true;
			} else {
				bsplit = false;
			}

			treeptr.dpvalEdgeFull = new double[treeptr.numchildren][0];
			treeptr.ddiffEdgeFull = new double[treeptr.numchildren][0];
			treeptr.dexpectEdgeFull = new double[treeptr.numchildren][0];
			treeptr.tsSigTFFull = (TreeSet<SigTFRecv2>[]) new TreeSet[treeptr.numchildren];

			if ((treeptr.numchildren == 1) && (treeptr != rootptr)) {
				boolean bfound = false;
				int nindex = 0;
				while (!bfound) {
					if (treeptr.parent.nextptr[nindex] == treeptr) {
						bfound = true;
						if (BDEBUG) {
							System.out.println(treeptr + "\t" + treeptr.parent
									+ "\t" + treeptr.parentptrA + "\t"
									+ treeptr.parent.dpvalEdgeFull);
						}

						if (treeptr.parentptrA == null) {
							treeptr.dpvalEdgeFull[0] = treeptr.parent.dpvalEdgeFull[nindex];
							treeptr.ddiffEdgeFull[0] = treeptr.parent.ddiffEdgeFull[nindex];
							treeptr.dexpectEdgeFull[0] = treeptr.parent.dexpectEdgeFull[nindex];
							treeptr.tsSigTFFull[0] = treeptr.parent.tsSigTFFull[nindex];
						}
					} else {
						nindex++;
					}
				}
			} else {
				treeptr.dpvalEdgeFull = new double[treeptr.numchildren][numrowstable];
				treeptr.ddiffEdgeFull = new double[treeptr.numchildren][numrowstable];
				treeptr.dexpectEdgeFull = new double[treeptr.numchildren][numrowstable];
				for (int nindex = 0; nindex < treeptr.numchildren; nindex++) {
					treeptr.tsSigTFFull[nindex] = new TreeSet<SigTFRecv2>(
							new SigTFRecv2Compare());
				}

				for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
					// System.out.println("---------------------------"+treeptr.ndepth);

					int nrowindex = 0;
					for (int nrow = 0; nrow < bindingData.regNames.length; nrow++) {
						int nsumthispath = 0;
						nsumthispath = treeptr.nextptr[nchild].numPath;

						for (int nel = 0; nel < bindingData.signedBindingValuesSorted.length; nel++) {
							if (bindingData.signedBindingValuesSorted[nel] != 0) {
								int nsumfullel = filteredClassifier.nBaseCount[nrow][bindingData.signedBindingValuesSorted[nel]
										+ filteredClassifier.noffset];
								double dexpectfull = (nsumthispath * nsumfullel)
										/ (double) ntotalcombined;
								treeptr.dexpectEdgeFull[nchild][nrowindex] = dexpectfull;

								double ddiffull = treeptr.ncountvals[nrow][nchild][nel]
										- dexpectfull;
								treeptr.ddiffEdgeFull[nchild][nrowindex] = ddiffull;

								treeptr.dpvalEdgeFull[nchild][nrowindex] = StatUtil
										.hypergeometrictail(
												treeptr.ncountvals[nrow][nchild][nel] - 1,
												nsumfullel, ntotalcombined
														- nsumfullel,
												nsumthispath);
								// assuming 0 always unique input
								if (bindingData.signedBindingValuesUnsorted
										.size() == 2) {
									treeptr.tsSigTFFull[nchild]
											.add(new SigTFRecv2(
													bindingData.regNames[nrow],
													treeptr.dpvalEdgeFull[nchild][nrowindex]));
								} else {
									treeptr.tsSigTFFull[nchild]
											.add(new SigTFRecv2(
													bindingData.regNames[nrow]
															+ " "
															+ bindingData.signedBindingValuesSorted[nel],
													treeptr.dpvalEdgeFull[nchild][nrowindex]));
								}

								if (bsplit) {
									int nsumthisel = treeptr.ncountTotals[nrow][nel];

									int nsumthisparentpath = treeptr.numPath;

									double dexpect = (nsumthispath * nsumthisel)
											/ (double) nsumthisparentpath;
									treeptr.dexpectEdgeSplit[nchild][nrowindex] = dexpect;
									double ddiff = treeptr.ncountvals[nrow][nchild][nel]
											- dexpect;
									treeptr.ddiffEdgeSplit[nchild][nrowindex] = ddiff;
									treeptr.dpvalEdgeSplit[nchild][nrowindex] = StatUtil
											.hypergeometrictail(
													treeptr.ncountvals[nrow][nchild][nel] - 1,
													nsumthisel,
													nsumthisparentpath
															- nsumthisel,
													nsumthispath);

									treeptr.dratioSplit[nchild][nrowindex] = (double) treeptr.ncountvals[nrow][nchild][nel]
											/ treeptr.ncountTotals[nrow][nel];

									if (bindingData.signedBindingValuesUnsorted
											.size() == 2) {
										int upregulatedChild = 0;
										if (treeptr.nextptr[0].dmean < treeptr.nextptr[1].dmean) {
											upregulatedChild = 1;
										}
										
										//TODO: what does this code do MARCEL
										String name = bindingData.regNames[nrow];
										if (bindingData.regTypes[nrow] == RegulatorBindingData.MIRNA
												&& treeptr.tranC.traindataTF[nrow].length != 0
												&& treeptr.tranC.traindataTF[nrow][0] != 0) {
											if (reg2DataSetIndex
															.get(name
																	.toUpperCase()) != null) {
												treeptr.tsSigTFEdgeSplit[nchild]
														.add(new SigTFRecv2(
																name,
																treeptr.dpvalEdgeSplit[nchild][nrowindex],
																treeptr.dratioSplit[nchild][nrowindex]));
											}

										} else {
											if (bindingData.regTypes[nrow] == RegulatorBindingData.TF) {
												treeptr.tsSigTFEdgeSplit[nchild]
														.add(new SigTFRecv2(
																name,
																treeptr.dpvalEdgeSplit[nchild][nrowindex],
																treeptr.dratioSplit[nchild][nrowindex]));
											}
										}
									} else {
										treeptr.tsSigTFEdgeSplit[nchild]
												.add(new SigTFRecv2(
														bindingData.regNames[nrow]
																+ " "
																+ bindingData.signedBindingValuesSorted[nel],
														treeptr.dpvalEdgeSplit[nchild][nrowindex],
														treeptr.dratioSplit[nchild][nrowindex]));
									}
								}
								nrowindex++;
							}
						}
					}
				}
			}

			for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
				computeStats(treeptr.nextptr[nchild], rootptr);
			}
		}
	}

	// ///////////////////////////////////////////////////////////////////////////////
	/**
	 * Record used in the ranking of transcription factors
	 */
	static class SigTFRecv2 {
		String szname;
		double dpval;
		double dpercent;

		SigTFRecv2(String szname, double dpval, double dpercent) {
			this.szname = szname;
			this.dpval = dpval;
			this.dpercent = dpercent;
		}

		SigTFRecv2(String szname, double dpval) {
			this.szname = szname;
			this.dpval = dpval;
			this.dpercent = -1;
		}
	}

	/**
	 * Compares TFs first based on the value first lower dpval, then great
	 * dpercent, then name
	 */
	static class SigTFRecv2Compare implements Comparator<SigTFRecv2> {
		public int compare(SigTFRecv2 cr1, SigTFRecv2 cr2) {
			if (cr1.dpval < cr2.dpval)
				return -1;
			else if (cr1.dpval > cr2.dpval)
				return 1;
			else if (cr1.dpercent > cr2.dpercent)
				return -1;
			else if (cr1.dpercent < cr2.dpercent)
				return 1;
			else
				return cr1.szname.compareTo(cr2.szname);

		}
	}

	// ///////////////////////////////////////////////////////////////////////////////////////
	/**
	 * Computes the fraction of combinations for vals that could lead to a
	 * greater difference than ddiff
	 */
	public double computepval(int[][] vals, double diff) {
		int[] rowSum = new int[vals[0].length];
		double dtimes = 0;
		int nsum0 = 0;
		int nsum1 = 0;

		for (int nindex = 0; nindex < vals[0].length; nindex++) {
			nsum0 += vals[0][nindex];
			nsum1 += vals[1][nindex];
			rowSum[nindex] = vals[0][nindex] + vals[1][nindex];
		}

		double dpvalsum = 0;
		int[] ncol0 = new int[vals[0].length];

		if (ncol0.length == 2) {
			for (ncol0[0] = 0; ncol0[0] <= Math.min(rowSum[0], nsum0); ncol0[0]++) {
				double dinnersum = 0;
				ncol0[1] = nsum0 - ncol0[0];

				if (ncol0[1] <= rowSum[1]) {
					double dweightsum0 = 0;
					double dweightsum1 = 0;
					for (int i = 0; i < ncol0.length; i++) {
						dweightsum0 += bindingData.signedBindingValuesSorted[i]
								* ncol0[i];
						dweightsum1 += bindingData.signedBindingValuesSorted[i]
								* (rowSum[i] - ncol0[i]);
					}
					double dcurrdiff = Math.abs(dweightsum0 / nsum0
							- dweightsum1 / nsum1);
					if (dcurrdiff >= diff) {
						dtimes++;
						dinnersum++;
					}
				}
				dpvalsum += StatUtil.hypergeometric(ncol0[0], nsum0, nsum1,
						rowSum[0])
						* dinnersum;
			}
		} else if (ncol0.length == 3) {
			for (ncol0[0] = 0; ncol0[0] <= Math.min(rowSum[0], nsum0); ncol0[0]++) {
				double dinnersum = 0;
				int nremainder0 = nsum0 - ncol0[0];
				int nremainder1 = nsum1 - (rowSum[0] - ncol0[0]);
				for (ncol0[1] = 0; ncol0[1] <= Math.min(rowSum[1], nremainder0); ncol0[1]++) {
					ncol0[2] = nsum0;
					for (int i = 0; i < ncol0.length - 1; i++) {
						ncol0[2] -= ncol0[i];
					}
					if (ncol0[2] <= rowSum[2]) {
						double dweightsum0 = 0;
						double dweightsum1 = 0;
						for (int i = 0; i < ncol0.length; i++) {
							dweightsum0 += bindingData.signedBindingValuesSorted[i]
									* ncol0[i];
							dweightsum1 += bindingData.signedBindingValuesSorted[i]
									* (rowSum[i] - ncol0[i]);
						}
						double dcurrdiff = Math.abs(dweightsum0 / nsum0
								- dweightsum1 / nsum1);
						if (dcurrdiff >= diff) {
							dtimes++;
							dinnersum += StatUtil.hypergeometric(ncol0[1],
									nremainder0, nremainder1, rowSum[1]);
						}
					}
				}
				dpvalsum += StatUtil.hypergeometric(ncol0[0], nsum0, nsum1,
						rowSum[0])
						* dinnersum;
			}
		} else if (ncol0.length == 1) {
			return 1;
		} else {
			throw new IllegalArgumentException(
					"can only handle two or three unique inputs");
		}

		return dpvalsum;
	}

	// ///////////////////////////////////////////////////////////////////////

	/**
	 * Recursively determines the best path through the model and its likelihood
	 */
	public double computevlogit(double[] vals, int[] pma,
			double[][][] theInstance, int[][][] theInstanceIndex, int ndepth,
			Treenode node, int[] bestpath, int instanceindex) throws Exception {
		double dmax = Double.NEGATIVE_INFINITY;
		int bestchild = 0;
		double dlogout = 0;
		double dval = 0;

		if (node.dsigma > 0) {
			if (pma[ndepth] != 0) {
				if (ndepth == 0) {
					dlogout = 0;
				} else {
					dlogout = Math.log(StatUtil.normaldensity(vals[ndepth],
							node.dmean, node.dsigma));
				}
			} else {
				dlogout = 0;
			}
		}

		if (node.numchildren == 0) {
			return dlogout;
		}

		int[] currbestpath = new int[bestpath.length];

		// log (a) + c(log(b_1) + log(b_2) + ... + log(b_n))
		// = log(a*(b_1*b_2*...*b_n)^c)

		double[] ptranlogit;

		if (BREGDREM) {
			ptranlogit = node.ptrans;
		} else if ((node.numchildren > 1) && (node.binit)) {
			ptranlogit = node.tranC.distributionForInstance(
					theInstanceIndex[node.ndepth][instanceindex],
					theInstance[node.ndepth][instanceindex]);
		} else {
			ptranlogit = CONSTANTA[node.numchildren];
		}

		for (int nchild = 0; nchild < node.numchildren; nchild++) {
			dval = dlogout;
			if (node.nextptr[nchild] != null) {
				double dvi = computevlogit(vals, pma, theInstance,
						theInstanceIndex, ndepth + 1, node.nextptr[nchild],
						bestpath, instanceindex);
				if (ptranlogit[nchild] == 0) {
					dval += Math.log(MINPROB);
				} else {
					dval += Math.log(ptranlogit[nchild]) + dvi;
				}
			}

			if (dval > dmax) {
				dmax = dval;
				bestchild = nchild;
				for (int nindex = 0; nindex < bestpath.length; nindex++) {
					currbestpath[nindex] = bestpath[nindex];
				}
			}
		}

		for (int nindex = 0; nindex < bestpath.length; nindex++) {
			bestpath[nindex] = currbestpath[nindex];
		}
		bestpath[ndepth] = bestchild;

		return dmax;
	}

	// ////////////////////////////////////////////////////////////////////////
	/**
	 * Evaluates the likelihood of the viterbi assignments of the holdoutdata
	 * under the current settings of the model parameters
	 */
	public double testhmmVHoldOut(Treenode treehmm) throws Exception {
		double dlike = 0;
		int[] bestpath = new int[traindata[0].length];
		for (int nrow = 0; nrow < holdoutdata.length; nrow++) {
			double dbest = computevlogit(holdoutdata[nrow], holdoutpma[nrow],
					holdoutpval, holdoutpvalIndex, 0, treehmm, bestpath, nrow);
			dlike += dbest;
		}
		return dlike;
	}

	// ////////////////////////////////////////////////////////////////////////
	/**
	 * Evaluates the likelihood of the viterbi assignments of the testdata under
	 * the current settings of the model parameters
	 */
	public double testhmmV(Treenode treehmm) throws Exception {
		double dlike = 0;
		int[] bestpath = new int[traindata[0].length];
		for (int nrow = 0; nrow < testdata.length; nrow++) {
			double dbest = computevlogit(testdata[nrow], testpma[nrow],
					testBinding.gene2RegBinding,
					testBinding.gene2RegBindingIndex, 0, treehmm, bestpath,
					nrow);
			dlike += dbest;
		}
		return dlike;
	}

	// ////////////////////////////////////////////////////////////////////////
	/**
	 * Evaluates the likelihood of the holdoutdata under the current settings of
	 * the model parameters
	 */
	public double testhmmHoldOut(Treenode treehmm) throws Exception {
		double dlike = 0;
		for (int nrow = 0; nrow < holdoutdata.length; nrow++) {
			backalg(holdoutdata[nrow], holdoutpma[nrow], holdoutpvalIndex,
					holdoutpval, 0, treehmm, nrow, false, nrow);
			if (treehmm.db <= 0) {
				dlike += Math.log(MINPROB);
				System.out.println("B warning row " + nrow + " is "
						+ treehmm.db);
			} else {
				dlike += Math.log(treehmm.db);
			}
		}
		return dlike;
	}

	// ////////////////////////////////////////////////////////////////////////
	/**
	 * Evaluates the likelihood of the testdata under the current settings of
	 * the model parameters
	 */
	public double testhmm(Treenode treehmm) throws Exception {
		double dlike = 0;
		for (int nrow = 0; nrow < testdata.length; nrow++) {
			backalg(testdata[nrow], testpma[nrow],
					testBinding.gene2RegBindingIndex,
					testBinding.gene2RegBinding, 0, treehmm, nrow, false, nrow);

			if (treehmm.db <= 0) {
				dlike += Math.log(MINPROB);
			} else {
				dlike += Math.log(treehmm.db);
			}
		}

		return dlike;
	}

	/**
	 * Returns the number of nodes in the tree pointed by root for which its
	 * ncurrtime field does not equal the ncurrtime parameter
	 */
	public int countNodes(Treenode root, int ncurrtime) {

		if (root == null) {
			return 0;
		} else {
			int nsum;
			if (root.ncurrtime != ncurrtime) {
				root.ncurrtime = ncurrtime;
				nsum = 1;
			} else {
				nsum = 0;
			}

			for (int nchild = 0; nchild < root.numchildren; nchild++) {
				nsum += countNodes(root.nextptr[nchild], ncurrtime);
			}

			return nsum;
		}
	}

	// ////////////////////////////////////////////////////////////////////////
	/**
	 * Implements a viterbi training method for the parameters with the current
	 * model structure
	 */
	public double trainhmmV(Treenode treehmm, boolean bpruneexempt)
			throws Exception {

		double dlike = 0;
		double doldlike;

		// first time through we will need to initialize the hmm without the
		// classifiers
		// then we will need to train the classifiers

		initAEV(treehmm);
		int[] bestpath = new int[traindata[0].length];

		double[] ptranlogit;

		for (int nrow = 0; nrow < traindata.length; nrow++) {
			double[] theInstance = trainBinding.gene2RegBinding[treehmm.ndepth][nrow];
			int[] theInstanceIndex = trainBinding.gene2RegBindingIndex[treehmm.ndepth][nrow];

			if (BREGDREM) {
				ptranlogit = treehmm.ptrans;
			} else if ((treehmm.numchildren > 1) && (treehmm.binit)) {
				ptranlogit = treehmm.tranC.distributionForInstance(
						theInstanceIndex, theInstance);
			} else {
				ptranlogit = CONSTANTA[treehmm.numchildren];
			}

			double dbest = computevlogit(traindata[nrow], trainpma[nrow],
					trainBinding.gene2RegBinding,
					trainBinding.gene2RegBindingIndex, 0, treehmm, bestpath,
					nrow);
			instanceAEV(treehmm, traindata[nrow], trainpma[nrow],
					trainBinding.gene2RegBinding[treehmm.ndepth][nrow],
					bestpath, Math.exp(dbest), nrow);

			dlike += dbest;

		}
		updateParams(treehmm, 0, traindata, trainpma, 0);

		if (BEQUALSTD) {
			averageChildrenSigmas(treehmm);
		}

		int ncount = 0;

		// very first time going to assign labels
		double dpredictitr;

		do {
			// train while there is still a big enough improvement
			doldlike = dlike;
			initAEV(treehmm);
			dlike = 0;

			for (int nrow = 0; nrow < traindata.length; nrow++) {
				double[] theInstance = trainBinding.gene2RegBinding[treehmm.ndepth][nrow];
				int[] theInstanceIndex = trainBinding.gene2RegBindingIndex[treehmm.ndepth][nrow];

				if (BREGDREM) {
					ptranlogit = treehmm.ptrans;
				} else if ((treehmm.numchildren > 1) && (treehmm.binit)) {
					ptranlogit = treehmm.tranC.distributionForInstance(
							theInstanceIndex, theInstance);
				} else {
					ptranlogit = CONSTANTA[treehmm.numchildren];
				}

				double dbest = computevlogit(traindata[nrow], trainpma[nrow],
						trainBinding.gene2RegBinding,
						trainBinding.gene2RegBindingIndex, 0, treehmm,
						bestpath, nrow);

				instanceAEV(treehmm, traindata[nrow], trainpma[nrow],
						trainBinding.gene2RegBinding[treehmm.ndepth][nrow],
						bestpath, Math.exp(dbest), nrow);

				dlike += dbest;

				if (Double.isNaN(dlike)) {
					traverse(treehmm, 0, true);

					System.out.println(nrow + "\t");
					for (int na = 0; na < trainBinding.gene2RegBinding[nrow].length; na++) {
						System.out.print(trainBinding.gene2RegBinding[nrow][na]
								+ "\t");
					}
					System.out.println();
					if (!BREGDREM) {
						for (int na = 0; na < treehmm.tranC.dcoeff.length; na++) {
							System.out.print(treehmm.tranC.dcoeff[na] + "\t");
						}
					}
					System.out.println();
					System.out.println("treehmm.db = " + treehmm.db);
					throw new Exception();
				}
			}

			updateParams(treehmm, 0, traindata, trainpma, 0);
			if (BEQUALSTD) {
				averageChildrenSigmas(treehmm);
			}

			ncount++;
			dpredictitr = (dbesttrainlike - dlike) / (dlike - doldlike);
		} while (((dlike - doldlike) / Math.abs(dlike) > BEPSILON)
				&& ((bpruneexempt) || (dpredictitr < MAXFUTUREITR)));

		if (BDEBUG) {
			System.out.println("Likelihood: " + dlike);
		}

		dtrainlike = dlike;

		return (dlike);
	}

	// ////////////////////////////////////////////////////////////////////////
	/**
	 * Implements a baum-welch method for training the parameters with the
	 * current model structure
	 */
	public double trainhmm(Treenode treehmm, boolean bpruneexempt)
			throws Exception {

		double dlike = 0;
		double doldlike;

		// first time through we will need to initialize the hmm without the
		// classifiers
		// then we will need to train the classifiers

		initAE(treehmm);
		nglobaliteration = 1;
		dlike = 0;
		double[] ptranlogit;

		traverse(treehmm, 0, false);
		for (int nrow = 0; nrow < traindata.length; nrow++) {
			treehmm.df = 1;

			double[] theInstance = trainBinding.gene2RegBinding[treehmm.ndepth][nrow];
			int[] theInstanceIndex = trainBinding.gene2RegBindingIndex[treehmm.ndepth][nrow];

			if (BREGDREM) {
				ptranlogit = treehmm.ptrans;
			} else if ((treehmm.numchildren > 1) && (treehmm.binit)) {
				ptranlogit = treehmm.tranC.distributionForInstance(
						theInstanceIndex, theInstance);
			} else {
				ptranlogit = CONSTANTA[treehmm.numchildren];
			}

			for (int nchild = 0; nchild < treehmm.numchildren; nchild++) {
				forwardalg(traindata[nrow], trainpma[nrow],
						trainBinding.gene2RegBindingIndex,
						trainBinding.gene2RegBinding, 1, ptranlogit[nchild],
						treehmm.nextptr[nchild], nrow, treehmm, nrow);
			}
			backalg(traindata[nrow], trainpma[nrow],
					trainBinding.gene2RegBindingIndex,
					trainBinding.gene2RegBinding, 0, treehmm, nrow, true, nrow);
			double dpj = treehmm.db;
			if (dpj == 0) {
				dlike += Math.log(MINPROB);
			} else {
				instanceAE(treehmm, 0, traindata[nrow], trainpma[nrow], dpj,
						nrow);
				dlike += Math.log(treehmm.db);
			}
		}

		traverse(treehmm, 0, false);
		updateParams(treehmm, 0, traindata, trainpma, 0);

		traverse(treehmm, 0, false);
		if (BEQUALSTD) {
			averageChildrenSigmas(treehmm);
		}

		int ncount = 0;

		// very first time going to assign labels
		double dpredictitr;

		do {
			nglobaliteration++;
			if (BDEBUG) {
				System.out.println("Global iteration is now "
						+ nglobaliteration);
			}

			// train while there is still a big enough improvement
			doldlike = dlike;
			initAE(treehmm);
			dlike = 0;

			for (int nrow = 0; nrow < traindata.length; nrow++) {
				double[] theInstance = trainBinding.gene2RegBinding[treehmm.ndepth][nrow];
				int[] theInstanceIndex = trainBinding.gene2RegBindingIndex[treehmm.ndepth][nrow];

				if (BREGDREM) {
					ptranlogit = treehmm.ptrans;
				} else if ((treehmm.numchildren > 1) && (treehmm.binit)) {
					ptranlogit = treehmm.tranC.distributionForInstance(
							theInstanceIndex, theInstance);
				} else {
					ptranlogit = CONSTANTA[treehmm.numchildren];
				}

				treehmm.df = 1;
				for (int nchild = 0; nchild < treehmm.numchildren; nchild++) {
					forwardalg(traindata[nrow], trainpma[nrow],
							trainBinding.gene2RegBindingIndex,
							trainBinding.gene2RegBinding, 1,
							ptranlogit[nchild], treehmm.nextptr[nchild], nrow,
							treehmm, nrow);
				}

				backalg(traindata[nrow], trainpma[nrow],
						trainBinding.gene2RegBindingIndex,
						trainBinding.gene2RegBinding, 0, treehmm, nrow, true,
						nrow);

				double dpj = treehmm.db;
         
				if (dpj == 0) {
					dlike += Math.log(MINPROB);
				} else {
					instanceAE(treehmm, 0, traindata[nrow], trainpma[nrow],
							dpj, nrow);
                                        //FIXME TESTNAN
                                        if (Double.isInfinite(treehmm.db)||Double.isNaN(treehmm.db)){
                                            dlike+=Math.log(MINPROB);
                                            //System.out.println(treehmm.db);
                                        }else{
                                            dlike += Math.log(treehmm.db);
                                        }
                                    
				}

				if (Double.isNaN(dlike)) {
					traverse(treehmm, 0, true);

					System.out.println(nrow + "\t");
					for (int na = 0; na < trainBinding.gene2RegBinding[treehmm.ndepth][nrow].length; na++) {
						System.out
								.print(trainBinding.gene2RegBinding[treehmm.ndepth][nrow][na]
										+ "\t");
					}
					System.out.println();
					if (!BREGDREM) {
						for (int na = 0; na < treehmm.tranC.dcoeff.length; na++) {
							System.out.print(treehmm.tranC.dcoeff[na] + "\t");
						}
					}
					System.out.println();
					System.out.println("treehmm.db = " + treehmm.db);
					throw new Exception();
				}
			}

			traverse(treehmm, 0, false);
			updateParams(treehmm, 0, traindata, trainpma, 0);
			if (BEQUALSTD) {
				averageChildrenSigmas(treehmm);
			}

			traverse(treehmm, 0, false);
			ncount++;
			dpredictitr = (dbesttrainlike - dlike) / (dlike - doldlike);
			if (BDEBUGMODEL) {
				System.out.println(ncount + " - Train hmm Likelihood: " + dlike
						+ " best " + dbesttrainlike + " " + "oldlike"
						+ doldlike + " " + dpredictitr);
			}
		} while (((dlike - doldlike) / Math.abs(dlike) > BEPSILON)
				&& ((bpruneexempt) || (dpredictitr < MAXFUTUREITR)));

		if (BDEBUG) {
			System.out.println("Likelihood: " + dlike);
		}

		dtrainlike = dlike;

		nglobaltime++;
		int nparams = countNodes(treehmm, nglobaltime);
		dgloballike = dlike;

		if (BDEBUG) {
			System.out.print(dlike + "\t" + nparams);
		}
		dlike -= nodepenalty * nparams;

		if (BDEBUG) {
			System.out.println("\t" + dlike);
		}

		return (dlike);
	}
}