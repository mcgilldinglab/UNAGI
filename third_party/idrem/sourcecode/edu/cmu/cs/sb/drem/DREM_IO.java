package edu.cmu.cs.sb.drem;

import edu.cmu.cs.sb.core.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.swing.event.*;
import java.util.*;
import java.text.*;
import java.net.*;
import java.util.jar.*;

/**
 * Class implementing the main input interface
 */
public class DREM_IO extends JFrame implements ActionListener, ChangeListener {
	/** IS DREM being run in a non-interactive, batch mode */
	boolean bbatchMode;
	/**
	 * The filename prefix used when saving the model and activities in batch
	 * mode
	 */
	String saveFile;

	String szorganismsourceval;
	String szxrefsourceval;
	boolean btraintest;
	public static final int GOANN = 0;
	public static final int XREF = 1;
	public static final int OBO = 2;
	public static final int MIRNAEXP = 3;
	public static final int FILETYPES = 4;
	boolean[] bdownloading = new boolean[FILETYPES];
	Object lockpd = new Object();
	Object lockxref = new Object();
	int[] npercentdone = new int[FILETYPES];
	boolean[] bexception = new boolean[FILETYPES];
	int nexceptions;
	static boolean bdisplaycurrent = false;
	static String EBIURL = "ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/";
	static String szDefaultFile = "";

	// Main
	static boolean ballowmergeDEF = false;
	static String szStaticFileDEF = "";
	static String szDataFileDEF = "";
	static String szInitFileDEF = "";
	static boolean bspotcheckDEF = false;
	static int nnormalizeDEF = 1;
	static String szGeneAnnotationFileDEF = "";
	static String szCrossRefFileDEF = "";
	static int ndbDEF = 0;
	static int nxrefDEF = 0;
	static int nstaticsourceDEF = 0;
	static int numchildDEF = 3;

	// Repeat
	static Vector<String> vRepeatFilesDEF = new Vector<String>();
	static boolean balltimeDEF = true;

	// Search Options
	static double dCONVERGENCEDEF = 0.01;
	static double dMinScoreDEF = 0.0;
	static double dDELAYPATHDEF = .15;
	static double dDMERGEPATHDEF = .15;
	static double dPRUNEPATHDEF = .15;
	static double dMINSTDDEVALDEF = 0.0;

	static double dMinScoreDIFFDEF = 0;
	static double dDELAYPATHDIFFDEF = 0;
	static double dDMERGEPATHDIFFDEF = 0;
	static double dPRUNEPATHDIFFDEF = 0;
	static int ninitsearchDEF = 0;
	static int nSEEDDEF = 1260;
	static double dNODEPENALTYDEF = 40;
	static boolean bPENALIZEDDEF = true;
	static boolean bstaticsearchDEF = true;

	// DREM with miRNA Data
	static String[] miRNAorganisms = { "User Provided", "Human", 
		"Rat", "Mouse", "Fruitfly", "Nematode",
		 };
	//TODO: write in correct names
	static String[] miRNAFiles = { "", 
			"human_miRNA_interactions.txt",
			"rat_miRNA_interactions.txt",
			"mouse_miRNA_interactions.txt",
			"fruitfly_miRNA_interactions.txt",
			"nematode_miRNA_interactions.txt",
			};
	static String miRNAInteractionDataFile = "";
	static String miRNAExpressionDataFile = "";
	static boolean checkStatusTF = true;
	// TODO Could default to true once DREM+miRNA code is merged in
	static boolean checkStatusmiRNA = true;
	static boolean miRNATakeLog = true;
	static boolean miRNAAddZero = false;
	static double miRNAWeight = 1.0;
	static double tfWeight = 0.5;
	static Vector<String> miRNARepeatFilesDEF = new Vector<String>();
	static boolean miRNAalltimeDEF = true;
	static boolean filtermiRNAExp = false;
	static boolean bgetmirnaexp = false;

	// DECOD Options
	static String fastaFile = "";
	static String decodPath = "";

	// Regulator Scoring
	static String regScoreFile = "";

	// Filtering
	static String szPrefilteredDEF = "";
	static int nMaxMissingDEF = 0;
	static double dMinExpressionDEF = 1;
	static double dMinCorrelationRepeatsDEF = 0;
	static boolean bfilterstaticDEF = false;

	// Gene Annotations
	static String sztaxonDEF = "";
	static String szevidenceDEF = "";
	static boolean bpontoDEF = true;
	static boolean bcontoDEF = true;
	static boolean bfontoDEF = true;
	static String szcategoryIDDEF = "";

	// GO Analysis
	static int nSamplesMultipleDEF = 500;
	static int nMinGoGenesDEF = 5;
	static int nMinGOLevelDEF = 3;
	static boolean brandomgoDEF = false;
	static boolean bmaxminDEF = false;// true;
	static double dpercentDEF = 0;

	// GUI
	static boolean brealXaxisDEF = false;
	static double dYaxisDEF = 1;
	static double dXaxisDEF = 1;
	static int nKeyInputTypeDEF = 1;
	static double dKeyInputXDEF = 3;
	static String SZSTATICDIR = "TFInput";
	static String szGeneOntologyFileDEF = "gene_ontology.obo";
	static double dnodekDEF = 1;
	
	// SDREM
	static double dProbBindingFunctional = 0.8;

	long s1;
	JRadioButton bfButton;
	JRadioButton fdrButton;
	JRadioButton noneButton;
	JRadioButton bfgoButton;
	JRadioButton randomgoButton;
	JRadioButton initopenButton;
	JRadioButton initsearchButton;
	JRadioButton initnoButton;

	JButton endSearchButton;
	JButton currentButton;
	static boolean bendsearch = false;

	JLabel statusLabel = new JLabel(" Starting Search... ");
	JLabel statusLabel15 = new JLabel(" ");
	JLabel statusLabel2 = new JLabel(" ");
	JLabel statusLabel3 = new JLabel(" ");
	JLabel statuscountLabel = new JLabel(" ");

	String szorig1val;
	String szstaticFileval;
	String szxrefval = szCrossRefFileDEF;
	String szorig2val;
	String szgoval = szGeneAnnotationFileDEF;
	String szgocategoryval = szGeneOntologyFileDEF;
	String szextraval;
	String szcategoryIDval;
	String szinitfileval;
	String sznumchildval;

	String szevidenceval;
	String sztaxonval;
	boolean ballowmergeval;
	boolean bpontoval;
	boolean bcontoval;
	boolean bfontoval;
	boolean bpenalizedmodelval;
	String szmaxmissingval;
	String szexpressval;
	String szfilterthresholdval;
	String szlbval;
	String szalphaval;
	String szpercentileval;
	String sznumberprofilesval;
	String szsamplepvalval;
	String szmingoval;
	String szmingolevelval;
	boolean bstaticcheckval;
	int ninitsearchval = 1;
	boolean brandomgoval = false;
	boolean btakelog = false;
	boolean bstaticsearchval;
	int ndb;
	int nxrefcb;
	int nstaticsourcecb;
	String szusergann;
	String szuserFileField;
	String szuserxref;
	String szepsilonval;
	String sznodepenaltyval;
	String szconvergenceval;
	String szminstddeval;
	String szprunepathval;
	String szdelaypathval;
	String szmergepathval;
	String szepsilonvaldiff;
	String szprunepathvaldiff;
	String szdelaypathvaldiff;
	String szmergepathvaldiff;
	String szseedval;
        String proteweightval="1.0";
        String epiweightval="1.0";
	boolean bmaxminval;

	int ntodownload;
	boolean balltimemiRNA = false;
	boolean balltime = false;
        boolean balltimeprote=false; //proteomics
        static boolean proteTakeLog = true; //proteomics
	static boolean proteAddZero = false; //proteomics
	boolean bspotincluded;
	boolean badd0 = false;
	// boolean bthreewaysplit;

	static JFileChooser theChooser = new JFileChooser();
	JLabel orig1Label;
	JLabel staticFileLabel;
	JLabel xrefLabel;
	JLabel xrefsourceLabel;
	JLabel categoryIDLabel;
	JLabel extraLabel;
	JLabel compare1Label;
	JLabel compare2Label;
	JLabel alphaLabel;
	JLabel corrmodelLabel;
	JLabel maxmissingLabel;
	JLabel filterthresholdLabel;
	JLabel expressLabel;
	JLabel maxchangeLabel;
	JLabel percentileLabel;
	JLabel lbLabel;
	JLabel samplepvalLabel;
	JLabel goLabel;
	JLabel mingoLabel;
	JLabel mingolevelLabel;
	JLabel randomgoLabel;
	JLabel epsilonLabel;
	JLabel prunepathLabel;
	JLabel delaypathLabel;
	JLabel mergeLabel;
	JLabel epsilonLabeldiff;
	JLabel prunepathLabeldiff;
	JLabel delaypathLabeldiff;
	JLabel mergeLabeldiff;
	JLabel numchildLabel;
	JLabel seedLabel;
	JLabel nodepenaltyLabel;
        JLabel pproteweightLabel; //proteomics weight label
        //JLabel pepigeneticWeightLabel; //epigenetics weight label
	JLabel initFileLabel;
	JLabel savedModelOptionsLabel;
	JLabel filterchoiceLabel;
	JLabel modelLabel;
	JLabel convergenceJLabel;
	JLabel minstddevJLabel;

	JComboBox orgcb;
	JComboBox xrefcb;
	JComboBox staticsourcecb;

	JRadioButton lognormButton, normButton, nonormButton;
	JRadioButton maxminButton, absButton;
	JRadioButton penalizedButton, traintestButton;

	ButtonGroup normGroup = new ButtonGroup();
	ButtonGroup modelGroup = new ButtonGroup();

	JCheckBox anncheck = new JCheckBox("Annotations", false);
	JCheckBox xrefcheck = new JCheckBox("Cross References", false);
	JCheckBox obocheck = new JCheckBox("Ontology", false);
	JCheckBox threewaycheck;
	JCheckBox allowmergecheck;
	JCheckBox staticsearchcheck;
	JCheckBox pcheck;
	JCheckBox fcheck;
	JCheckBox ccheck;
	JCheckBox spotcheck;
	JCheckBox staticcheck;
	JDialog executeDialognf = null;
        
        
	// miRNA Search //////////////////////////////////
	JComboBox miRNACB;
	JButton miRNAComboBoxHButton = new JButton(Util
			.createImageIcon("Help16.gif"));
	JTextField miRNAInteractionField;
	JButton staticmiRNAInteractionFileButton = new JButton("Browse...", Util
			.createImageIcon("Open16.gif"));
	JButton staticmiRNAInterHButton = new JButton(Util
			.createImageIcon("Help16.gif"));

	JTextField miRNAExpressionField;
	JButton staticmiRNAExpressionFileButton = new JButton("Browse...", Util
			.createImageIcon("Open16.gif"));
	JButton staticmiRNAExpHButton = new JButton(Util
			.createImageIcon("Help16.gif"));

	ListDialog miRNARepeatList;
	JButton miRNARepeatHButton = new JButton(Util.createImageIcon("Help16.gif"));

	ButtonGroup miRNAnormGroup = new ButtonGroup();
	JRadioButton miRNAlognormButton;
	JRadioButton miRNAnormButton;
	JRadioButton miRNAnonormButton;
	JButton miRNANormHButton = new JButton(Util.createImageIcon("Help16.gif"));

	static JCheckBox TfCheckBox;
	static JCheckBox miRNACheckBox;
	JButton miRNAScoringHButton = new JButton(Util
			.createImageIcon("Help16.gif"));

	JCheckBox filtermiRNABox;
	JButton miRNAFilterHButton = new JButton(Util.createImageIcon("Help16.gif"));

	JSpinner spinnerMIRNAWeight;
	JButton miRNAWeightHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JSpinner spinnerTFWeight;
	JButton miRNATFWeightHButton = new JButton(Util
			.createImageIcon("Help16.gif"));
	JSpinner spinnerdProbBind;
	
	// FIXME Proteomics GUI
        static String djProteFile = "";
        static String djPPIFile="";
        JButton ProteFileButton = new JButton("Browse...",
                                Util.createImageIcon("Open16.gif"));
        ListDialog ProteRepeatList;
        JButton ProteRepeatButton = new JButton ("Repeat Proteomics Data File...", 
                                Util.createImageIcon("Open16.gif"));
        
        
        JButton ProteHButton = new JButton (Util.createImageIcon("Help16.gif"));
        JButton ProteHCButton = new JButton (Util.createImageIcon("Help16.gif"));
        JButton ProteHRButton = new JButton (Util.createImageIcon("Help16.gif"));
        
        JButton ProtePPIButton= new JButton("Browse...",Util.createImageIcon("Open16.gif"));
        JButton ProteHPPIButton =new JButton (Util.createImageIcon("Help16.gif"));
        JTextField ProtePPIField;
        JLabel ProtePPILabel;
        
        JTextField ProteField;
      
        JLabel ProteFileLabel;
        JCheckBox ProteTFCheckBox;
        JCheckBox ProteAllCheckBox;
        JCheckBox ProteNoCheckBox;
        
        JRadioButton ProtelognormButton;
        JRadioButton ProtenormButton;
        JRadioButton ProtenonormButton;
        
        
        static boolean bProteTF = false;
        static boolean bProteAll = false;
        static boolean bProteNo =true;
        static Vector<String> ProteRepeatFilesDEF = new Vector<String>();
        static boolean ProtealltimeDEF = true;
       
        
        // TOIMPLEMENT adds the methylation function panel
        static String djMethyFile = ""; // Methylation file 
        static String djMethyGTF=""; // TF motif file for Methylation data
        
        // MethyFile Button
        JButton MethyFileButton = new JButton("Browse...",
                                Util.createImageIcon("Open16.gif"));
        
        JButton MethyHButton = new JButton (Util.createImageIcon("Help16.gif"));
        JTextField MethyField;
        JLabel MethyLabel;
        
        // MethyTFMotif Button 
        JButton MethyGTFButton = new JButton ("Browse...",Util.createImageIcon("Open16.gif"));
        JButton MethyGTFHButton = new JButton (Util.createImageIcon("Help16.gif"));
        JTextField MethyGTFField;
        JLabel MethyGTFLabel;
        
        //
        
	// DECOD gui
	JButton fastaDataFileButton = new JButton("Browse...", Util
			.createImageIcon("Open16.gif"));
	JButton fastaDataHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JTextField fastaDataField;
	JButton decodPathButton = new JButton("Browse...", Util
			.createImageIcon("Open16.gif"));
	JButton decodPathHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JTextField decodPathField;

	// Regulator Scoring GUI
	JButton regScoreFileButton = new JButton("Browse...", Util
			.createImageIcon("Open16.gif"));
	JButton regScoreHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JTextField regScoreField;

	static int NUMCOLS = 42;

	// Strings for the labels
	static Color lightBlue = new Color(190, 255, 190);
	static Color defaultColor;

	static String[] staticsourceArray = { "User provided" };

	static String szorig1 = "Expression Data File: ";
	static String szstaticFile = "TF-gene Interactions File: ";
	static String szgo = "Gene Annotation File: ";
	static String szxreflt = "Cross Reference File: ";
	static String szxrefsource = "Cross Reference Source: ";
	static String szmaxmissing = "Maximum Number of Missing Values: ";
	static String szstaticcheck = "Filter gene if it has no transcription factor input data ";
	static String szextra = "Pre-filtered Gene File: ";
	static String szexpress = "Minimum Absolute Expression Change: ";
	static String szfilterthreshold = "Minimum Correlation between Repeats: ";
	static String szlb = "Minimum Correlation: ";
	static String szpercentile = "Minimum Correlation Percentile (repeat only):";
	static String szalpha = "Significance Level: ";
	static String szcorrmodel = "Maximum Correlation: ";
	static String szsamplepval = "Number of samples for randomized multiple hypothesis correction: ";
	static String szmingo = "Minimum number of genes: ";
	static String szmingolevel = "Minimum GO level: ";
	static String szrandomgo = "Multiple hypothesis correction method: ";
	static String szcategoryID = "Category ID mapping file: ";
	static String szspotcheck = "Spot IDs in the data file";
	static String sznumchild = "Maximum number of paths out of a split: ";
	static String szstaticsearch = "Use regulator-gene interaction data to build model ";
	static String szallowmerge = "Allow Path Merges";

	static String szepsilon = "Train-Test Main search score %: ";
	static String szprunepath = "Train-Test Delete path score %: ";
	static String szdelaypath = "Train-Test Delay spit score %: ";
	static String szmergepath = "Train-Test Merge path score %: ";
	static String szepsilondiff = "difference threshold (>= 0): ";
	static String szprunepathdiff = "difference threshold (<= 0): ";
	static String szdelaypathdiff = "difference threshold (<= 0): ";
	static String szmergepathdiff = "difference threshold (<= 0): ";
	static String szseed = "Train-Test Random Seed: ";
	static String sznodepenalty = "Penalized Likelihood Node Penalty: ";
	static String szinitfile = "Saved Model File: ";
	static String szsavedmodeloption = "Saved Model: ";
	static String szfilterchoice = "Change should be based on:";
	static String szconvergence = "Convergence Likelihood %";
	static String szminstddev = "Minimum Standard Deviation: ";

	JTextField orig1Field;
	JTextField staticFileField;
	JTextField goField;
	JTextField extraField;
	JTextField xrefField;
	JTextField categoryIDField;
	JTextField taxonField;
	JTextField evidenceField;
	JTextField initFileField;

	JButton orig1Button = new JButton("Browse...", Util
			.createImageIcon("Open16.gif"));
	JButton orig2Button = new JButton("Browse...", Util
			.createImageIcon("Open16.gif"));
	JButton staticFileButton = new JButton("Browse...", Util
			.createImageIcon("Open16.gif"));
	JButton goLabelButton = new JButton("Browse...", Util
			.createImageIcon("Open16.gif"));
	JButton xrefButton = new JButton("Browse...", Util
			.createImageIcon("Open16.gif"));
	JButton extraButton = new JButton("Browse...", Util
			.createImageIcon("Open16.gif"));
	JButton categoryIDButton = new JButton("Browse...", Util
			.createImageIcon("Open16.gif"));
	JButton initfileButton = new JButton("Browse...", Util
			.createImageIcon("Open16.gif"));

	JButton clusterAButton; //execute button
	JButton staticFileHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton logHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton viewButton = new JButton("View Expression Data");
	JButton viewstaticButton = new JButton("View TF-gene Data");
	JButton staticsourceHButton = new JButton(Util
			.createImageIcon("Help16.gif"));
	JButton repeatButton = new JButton("Repeat Data...", Util
			.createImageIcon("Open16.gif"));
	JButton miRNARepeatButton = new JButton("miRNA Repeat Data...", Util
			.createImageIcon("Open16.gif"));
	JButton orig1HButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton orig2HButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton advancedHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton goLabelHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton categoryIDHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton numchildHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton extraHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton numberProfileHButton = new JButton(Util
			.createImageIcon("Help16.gif"));
	JButton changeHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton maxmissingHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton filterthresholdHButton = new JButton(Util
			.createImageIcon("Help16.gif"));
	JButton expressHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton corrmodelHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton alphamodelHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton lbHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton percentileHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton samplegeneHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton samplemodelHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton samplepvalHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton mingoHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton mingolevelHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton methodHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton allowmergeHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton xrefsourceHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton downloadlistgoHButton = new JButton(Util
			.createImageIcon("Help16.gif"));
	JButton xrefHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton presetsHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton executeHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton spotHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton infoButton = new JButton(Util.createImageIcon("About16.gif"));
	JButton viewHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton viewstaticHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton evidenceHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton ontoHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton taxonHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton randomgoHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton threewayHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton epsilonHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton prunepathHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton delaypathHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton mergepathHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton seedHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton staticHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton initfileHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton initsearchHButton = new JButton(Util.createImageIcon("Help16.gif"));
	JButton staticsearchHButton = new JButton(Util
			.createImageIcon("Help16.gif"));
	JButton filterchoiceHButton = new JButton(Util
			.createImageIcon("Help16.gif"));
	JButton nodepenaltyHButton = new JButton(Util.createImageIcon("Help16.gif"));
        JButton proteweightHButton = new JButton(Util.createImageIcon("Help16.gif")); // proteomics weight help button
        JButton epigeneticweightHButton = new JButton(Util.createImageIcon("Help16.gif")); // epigenetic weight help button
	JButton modelframeworkHButton = new JButton(Util
			.createImageIcon("Help16.gif"));
	JButton convergencePanelHButton = new JButton(Util
			.createImageIcon("Help16.gif"));
	JButton minstddevPanelHButton = new JButton(Util
			.createImageIcon("Help16.gif"));

	JButton optionsButton = new JButton("Options...", Util
			.createImageIcon("Preferences16.gif"));
	static JFileChooser fc = new JFileChooser();

	JSpinner thespinnermingo;
	JSpinner thespinnermingolevel;
	JSpinner thespinnermaxmissing;
	JSpinner thespinnerfilterthreshold;
	JSpinner thespinnerexpress;
	JSpinner thespinnersamplepval;
	JSpinner thespinnerepsilon;
	JSpinner thespinnerprunepath;
	JSpinner thespinnerdelaypath;
	JSpinner thespinnermergepath;
	JSpinner thespinnerseed;
	JSpinner thespinnernumchild;
	JSpinner thespinnerconvergence;
	JSpinner thespinnerminstddev;

	JSpinner thespinnernodepenalty;
        JSpinner theproteWeight; //Proteomics weight
        //JSpinner theepigeneticWeight; //epigenetics weight
	JSpinner thespinnerepsilondiff;
	JSpinner thespinnerprunepathdiff;
	JSpinner thespinnerdelaypathdiff;
	JSpinner thespinnermergepathdiff;

	JCheckBox thelogcheck;
	JDialog theOptions;
	ListDialog theRepeatList;
	String szClusterA = "Execute";

	/**
	 * Class constructor - builds the input interface and run based on user-inputs
         * calls parseDefaults to get the initial settings from a default settings file if specified
	 */
	public DREM_IO() throws FileNotFoundException, IOException {
		super("iDREM -interactive Dynamic Regulatory Events Miner");
                
                File dir = new File(SZSTATICDIR);
                {
                        String[] children = dir.list();
                        if (children == null) {
                                // Either dir does not exist or is not a directory
                                final JFrame fframe = this;
                                javax.swing.SwingUtilities.invokeLater(new Runnable() {
                                        public void run() {
                                                JOptionPane.showMessageDialog(fframe, "The directory "
                                                                + SZSTATICDIR + " was not found.",
                                                                "Directory not found",
                                                                JOptionPane.WARNING_MESSAGE);
                                        }
                                });
                        } else {
                                staticsourceArray = new String[children.length + 1];
                                staticsourceArray[0] = "User Provided";
                                for (int i = 0; i < children.length; i++) {
                                        // Get filename of file or directory
                                        staticsourceArray[i + 1] = children[i];
                                }
                        }
                }
     
                /*
                SZSTATICDIR=this.getClass().getClassLoader().getResource(SZSTATICDIR).getPath();
                File dir = new File(SZSTATICDIR);
		{
			String[] children = dir.list();
			if (children == null) {
				// Either dir does not exist or is not a directory
				final JFrame fframe = this;
				javax.swing.SwingUtilities.invokeLater(new Runnable() {
					public void run() {
						JOptionPane.showMessageDialog(fframe, "The directory "
								+ SZSTATICDIR + " was not found.",
								"Directory not found",
								JOptionPane.WARNING_MESSAGE);
					}
				});
			} else {
				staticsourceArray = new String[children.length + 1];
				staticsourceArray[0] = "User Provided";
				for (int i = 0; i < children.length; i++) {
					// Get filename of file or directory
					staticsourceArray[i + 1] = children[i];
				}
			}
		}
                 */
                
                
		staticsourcecb = new JComboBox(staticsourceArray);
		staticsourcecb.addActionListener(this);

		bbatchMode = false;
		saveFile = "";
                  
                /*
                //REPLACED BY DREM_IO_Batch
		try {
			parseDefaults();
		} catch (IllegalArgumentException iex) {
			final IllegalArgumentException fiex = iex;
			final JFrame fframe = this;
			javax.swing.SwingUtilities.invokeLater(new Runnable() {
				public void run() {
					JOptionPane.showMessageDialog(fframe, fiex.getMessage(),
							"Exception thrown", JOptionPane.ERROR_MESSAGE);
				}
			});
		} catch (Exception ex) {
			final Exception fex = ex;
			final JFrame fframe = this;
			javax.swing.SwingUtilities.invokeLater(new Runnable() {
				public void run() {
					JOptionPane.showMessageDialog(fframe, fex.toString(),
							"Exception thrown", JOptionPane.ERROR_MESSAGE);
					fex.printStackTrace(System.out);
				}
			});
		}
                */
                
		orig1Label = new JLabel(szorig1, JLabel.TRAILING);
		staticFileLabel = new JLabel(szstaticFile, JLabel.TRAILING);
		extraLabel = new JLabel(szextra, JLabel.TRAILING);
		initFileLabel = new JLabel(szinitfile, JLabel.TRAILING);
		xrefLabel = new JLabel(szxreflt, JLabel.TRAILING);
		alphaLabel = new JLabel(szalpha, JLabel.TRAILING);
		lbLabel = new JLabel(szlb, JLabel.TRAILING);
		maxmissingLabel = new JLabel(szmaxmissing, JLabel.TRAILING);
		filterthresholdLabel = new JLabel(szfilterthreshold, JLabel.TRAILING);
		bmaxminval = bmaxminDEF;
		expressLabel = new JLabel(szexpress, JLabel.TRAILING);
		savedModelOptionsLabel = new JLabel(szsavedmodeloption, JLabel.TRAILING);

		corrmodelLabel = new JLabel(szcorrmodel, JLabel.TRAILING);
		percentileLabel = new JLabel(szpercentile, JLabel.TRAILING);

		samplepvalLabel = new JLabel(szsamplepval, JLabel.TRAILING);
		goLabel = new JLabel(szgo, JLabel.TRAILING);
		categoryIDLabel = new JLabel(szcategoryID, JLabel.TRAILING);

		categoryIDField = new JTextField(szcategoryIDDEF, JLabel.TRAILING);
		categoryIDField.setColumns(NUMCOLS - 2);

		initFileField = new JTextField(szInitFileDEF, JLabel.TRAILING);
		initFileField.setColumns(NUMCOLS - 2);

		orig1Field = new JTextField(szDataFileDEF, JLabel.TRAILING);
		orig1Field.setColumns(NUMCOLS);

		staticFileField = new JTextField(szStaticFileDEF, JLabel.TRAILING);
		staticFileField.setColumns(NUMCOLS);

		xrefField = new JTextField(szCrossRefFileDEF, JLabel.TRAILING);
		xrefField.setColumns(NUMCOLS);

		goField = new JTextField(szGeneAnnotationFileDEF, JLabel.TRAILING);
		goField.setColumns(NUMCOLS);

		extraField = new JTextField(szPrefilteredDEF, JLabel.TRAILING);
		extraField.setColumns(NUMCOLS);

		mingoLabel = new JLabel(szmingo, JLabel.TRAILING);
		mingolevelLabel = new JLabel(szmingolevel, JLabel.TRAILING);

		downloadlistgoHButton.addActionListener(this);
		presetsHButton.addActionListener(this);
		executeHButton.addActionListener(this);

		Container contentPane = getContentPane();

		BoxLayout layout = new BoxLayout(contentPane, BoxLayout.Y_AXIS);
		contentPane.setLayout(layout);
		contentPane.setBackground(lightBlue);

		JPanel pstaticsource = new JPanel(new SpringLayout());
		pstaticsource.setBackground(lightBlue);

		pstaticsource.add(new JLabel("TF-gene Interaction Source: "));
		pstaticsource.add(staticsourcecb);
		pstaticsource.add(staticsourceHButton);
		staticsourceHButton.addActionListener(this);
		staticsourcecb.setSelectedIndex(nstaticsourceDEF);
		handlestaticsource();
		SpringUtilities.makeCompactGrid(pstaticsource, 1, 3, 0, 2, 0, 3);

		contentPane.add(pstaticsource);
		JPanel p = new JPanel(new SpringLayout());
		JLabel step1 = new JLabel("  1.  Data Input:");

		p.add(step1);
		p.add(new JLabel(""));
		p.add(new JLabel(""));
		p.add(new JLabel(""));
		p.setBackground(lightBlue);

		p.add(new JLabel(""));
		p.add(pstaticsource);
		p.add(new JLabel(""));
		p.add(new JLabel(""));

		p.add(staticFileLabel);
		p.add(staticFileField);
		p.add(staticFileButton);
		p.add(staticFileHButton);

		staticFileButton.addActionListener(this);
		staticFileHButton.addActionListener(this);

		p.add(orig1Label);
		p.add(orig1Field);
		p.add(orig1Button);
		p.add(orig1HButton);

		orig1Button.addActionListener(this);
		orig1HButton.addActionListener(this);

		p.add(initFileLabel);
		p.add(initFileField);
		p.add(initfileButton);
		p.add(initfileHButton);
		initfileButton.addActionListener(this);
		initfileHButton.addActionListener(this);

		repeatButton.setPreferredSize(new Dimension(175, 28));
		repeatButton.setMinimumSize(new Dimension(175, 28));
		repeatButton.setMaximumSize(new Dimension(175, 28));
		viewButton.setPreferredSize(new Dimension(175, 28));
		viewButton.setMinimumSize(new Dimension(175, 28));
		viewButton.setMaximumSize(new Dimension(175, 28));
		viewstaticButton.setPreferredSize(new Dimension(175, 28));
		viewstaticButton.setMinimumSize(new Dimension(175, 28));
		viewstaticButton.setMaximumSize(new Dimension(175, 28));

		JPanel pexpress = new JPanel(new SpringLayout());
		JPanel viewpanel = new JPanel(new SpringLayout());
		p.add(new JLabel(""));
		viewpanel.add(viewstaticButton);
		viewpanel.add(viewstaticHButton);
		viewpanel.add(viewButton);
		viewButton.addActionListener(this);
		viewstaticButton.addActionListener(this);
		viewpanel.add(viewHButton);

		viewHButton.addActionListener(this);
		viewstaticHButton.addActionListener(this);
		spotHButton.addActionListener(this);
		repeatButton.addActionListener(this);
		defaultColor = repeatButton.getBackground();
		if (DREM_IO.vRepeatFilesDEF.size() >= 1) {
			repeatButton.setBackground(ListDialog.buttonColor);
		}

		orig2HButton.addActionListener(this);
		viewpanel.setBackground(lightBlue);

		spotcheck = new JCheckBox(szspotcheck, bspotcheckDEF);
		spotcheck.setBackground(lightBlue);
		spotcheck.setHorizontalAlignment(JCheckBox.TRAILING);

		viewpanel.add(spotcheck);
		viewpanel.add(spotHButton);
		viewpanel.add(repeatButton);
		viewpanel.add(orig2HButton);
		SpringUtilities.makeCompactGrid(viewpanel, 2, 4, 11, 2, 5, 3);
		pexpress.add(viewpanel);

		lognormButton = new JRadioButton("Log normalize data");
		normButton = new JRadioButton("Normalize data");
		lognormButton.setBackground(lightBlue);
		normButton.setBackground(lightBlue);
		nonormButton = new JRadioButton("No normalization/add 0");
		nonormButton.setBackground(lightBlue);
		JPanel normPanel = new JPanel();
		if (nnormalizeDEF == 0) {
			lognormButton.setSelected(true);
		} else if (nnormalizeDEF == 1) {
			normButton.setSelected(true);
		} else {
			nonormButton.setSelected(true);
		}

		normPanel.add(lognormButton);
		normPanel.add(normButton);
		normPanel.add(nonormButton);
		normGroup.add(lognormButton);
		normGroup.add(normButton);
		normGroup.add(nonormButton);

		logHButton.addActionListener(this);
		normPanel.add(logHButton);
		pexpress.add(normPanel);
		p.add(pexpress);
		pexpress.setBackground(lightBlue);
		SpringUtilities.makeCompactGrid(pexpress, 2, 1, 0, 0, 0, 0);
		p.add(new JLabel(""));
		p.add(new JLabel(""));
		normPanel.setBackground(lightBlue);

		JPanel pf21 = new JPanel();
		pf21.setBackground(Color.white);
		JPanel pf22 = new JPanel();
		pf22.setBackground(Color.white);
		JPanel pf23 = new JPanel();
		pf23.setBackground(Color.white);
		JPanel pf24 = new JPanel();
		pf24.setBackground(Color.white);
		p.add(pf21);
		p.add(pf22);
		p.add(pf23);
		p.add(pf24);
		JLabel step2 = new JLabel("  2.  Gene Annotation Input:");

		p.add(step2);
		p.add(new JLabel(""));
		p.add(new JLabel(""));
		p.add(new JLabel(""));

		JPanel psource = new JPanel(new SpringLayout());
		psource.setBackground(lightBlue);

		orgcb = new JComboBox(GoAnnotations.organisms);
		orgcb.setSelectedIndex(ndbDEF);
		orgcb.addActionListener(this);

		psource.add(new JLabel("Gene Annotation Source: "));
		psource.add(orgcb);
		psource.add(presetsHButton);

		xrefcb = new JComboBox(GoAnnotations.defaultxrefs);
		xrefcb.addActionListener(this);

		psource.add(new JLabel(szxrefsource));
		psource.add(xrefcb);
		psource.add(xrefsourceHButton);
		xrefsourceHButton.addActionListener(this);
		handlendbval();
		xrefcb.setSelectedIndex(nxrefDEF);
		handlenxrefval();
		SpringUtilities.makeCompactGrid(psource, 2, 3, 0, 2, 0, 3);
		p.add(new JLabel(""));
		p.add(psource);

		p.add(new JLabel(""));
		p.add(new JLabel(""));

		p.add(goLabel);
		p.add(goField);
		p.add(goLabelButton);
		p.add(goLabelHButton);
		goLabelButton.addActionListener(this);
		goLabelHButton.addActionListener(this);

		p.setBackground(lightBlue);
		p.add(xrefLabel);
		p.add(xrefField);
		p.add(xrefButton);
		p.add(xrefHButton);
		xrefButton.addActionListener(this);
		xrefHButton.addActionListener(this);

		JPanel pdownload = new JPanel();
		pdownload.setBackground(lightBlue);
		JLabel dlabel = new JLabel("Download the latest:", JLabel.TRAILING);
		pdownload.add(dlabel);

		anncheck.setBackground(lightBlue);
		xrefcheck.setBackground(lightBlue);
		obocheck.setBackground(lightBlue);
		pdownload.add(anncheck);
		pdownload.add(xrefcheck);
		pdownload.add(obocheck);
		pdownload.add(downloadlistgoHButton);
		p.add(new JLabel(""));
		p.add(pdownload);
		p.add(new JLabel(""));
		p.add(new JLabel(""));

		JPanel pf31 = new JPanel();
		pf31.setBackground(Color.white);
		JPanel pf32 = new JPanel();
		pf32.setBackground(Color.white);
		JPanel pf33 = new JPanel();
		pf33.setBackground(Color.white);
		JPanel pf34 = new JPanel();
		pf34.setBackground(Color.white);
		p.add(pf31);
		p.add(pf32);
		p.add(pf33);
		p.add(pf34);

		JLabel step3 = new JLabel("  3.  Options:");
		p.add(step3);
		p.add(new JLabel(""));
		p.add(new JLabel(""));
		p.add(new JLabel(""));

		SpinnerNumberModel snmodelmingo = new SpinnerNumberModel(new Integer(
				nMinGoGenesDEF), new Integer(1), null, new Integer(1));
		thespinnermingo = new JSpinner(snmodelmingo);
		thespinnermingo.setPreferredSize(new Dimension(60, 24));
		thespinnermingo.setMaximumSize(new Dimension(60, 24));

		SpinnerNumberModel snmodelmingolevel = new SpinnerNumberModel(
				new Integer(nMinGOLevelDEF), new Integer(1), null, new Integer(
						1));
		thespinnermingolevel = new JSpinner(snmodelmingolevel);
		thespinnermingolevel.setPreferredSize(new Dimension(60, 24));
		thespinnermingolevel.setMaximumSize(new Dimension(60, 24));

		SpinnerNumberModel snmodelMissing = new SpinnerNumberModel(new Integer(
				nMaxMissingDEF), new Integer(0), null, new Integer(1));
		thespinnermaxmissing = new JSpinner(snmodelMissing);
		thespinnermaxmissing.setPreferredSize(new Dimension(60, 24));
		thespinnermaxmissing.setMaximumSize(new Dimension(60, 24));
		SpinnerNumberModel snmodelFilter = new SpinnerNumberModel(new Double(
				dMinCorrelationRepeatsDEF), new Double(-1.1), new Double(1.1),
				new Double(0.05));
		thespinnerfilterthreshold = new JSpinner(snmodelFilter);
		thespinnerfilterthreshold.setPreferredSize(new Dimension(60, 24));
		thespinnerfilterthreshold.setMaximumSize(new Dimension(60, 24));

		SpinnerNumberModel snmodelExpress = new SpinnerNumberModel(new Double(
				dMinExpressionDEF), new Double(-.05), null, new Double(0.05));
		thespinnerexpress = new JSpinner(snmodelExpress);
		thespinnerexpress.setPreferredSize(new Dimension(60, 24));
		thespinnerexpress.setMaximumSize(new Dimension(60, 24));

		SpinnerNumberModel snmodelpval = new SpinnerNumberModel(new Integer(
				nSamplesMultipleDEF), new Integer(1), null, new Integer(50));
		thespinnersamplepval = new JSpinner(snmodelpval);
		thespinnersamplepval.setPreferredSize(new Dimension(60, 24));
		thespinnersamplepval.setMaximumSize(new Dimension(60, 24));

		// Tell accessibility tools about label/textfield pairs.
		xrefLabel.setLabelFor(xrefField);
		orig1Label.setLabelFor(orig1Field);
		staticFileLabel.setLabelFor(staticFileField);
		extraLabel.setLabelFor(extraField);
		categoryIDLabel.setLabelFor(categoryIDField);
		maxmissingLabel.setLabelFor(thespinnermaxmissing);
		filterthresholdLabel.setLabelFor(thespinnerfilterthreshold);
		expressLabel.setLabelFor(thespinnerexpress);

		samplepvalLabel.setLabelFor(thespinnersamplepval);
		goLabel.setLabelFor(goField);
		mingoLabel.setLabelFor(thespinnermingo);
		mingolevelLabel.setLabelFor(thespinnermingolevel);

		JPanel padvanced = new JPanel();

		clusterAButton = new JButton(szClusterA);
		clusterAButton.setBackground(ListDialog.buttonColor);
		clusterAButton.addActionListener(this);
		clusterAButton.setPreferredSize(new Dimension(400, 30));
		clusterAButton.setMinimumSize(new Dimension(400, 30));
		clusterAButton.setMaximumSize(new Dimension(400, 30));

		optionsButton.setPreferredSize(new Dimension(250, 28));
		optionsButton.setMinimumSize(new Dimension(250, 28));
		optionsButton.setMaximumSize(new Dimension(250, 28));

		padvanced.add(optionsButton);
		padvanced.add(advancedHButton);
		padvanced.setBackground(lightBlue);

		advancedHButton.addActionListener(this);
		p.add(new JLabel(""));
		p.add(padvanced);
		p.add(new JLabel(""));
		p.add(new JLabel(""));  

		optionsButton.addActionListener(this);

		contentPane.add(Box.createRigidArea(new Dimension(0, 6)));
		clusterAButton.setAlignmentX(Component.CENTER_ALIGNMENT);

		JPanel pf41 = new JPanel();
		pf41.setBackground(Color.white);
		JPanel pf42 = new JPanel();
		pf42.setBackground(Color.white);
		JPanel pf43 = new JPanel();
		pf43.setBackground(Color.white);
		JPanel pf44 = new JPanel();
		pf44.setBackground(Color.white);
		p.add(pf41);
		p.add(pf42);
		p.add(pf43);
		p.add(pf44);

		p.add(new JLabel("  4.  Execute:"));
		p.add(new JLabel(""));
		p.add(new JLabel(""));
		p.add(new JLabel(""));
		p.add(new JLabel(""));

		JPanel cpanel = new JPanel();
		cpanel.setBackground(lightBlue);
		cpanel.add(clusterAButton);
		cpanel.add(executeHButton);
		p.add(cpanel);
		p.add(new JLabel(""));
		p.add(new JLabel(""));

		SpringUtilities.makeCompactGrid(p, 18, 4, 0, 2, 0, 2);
		contentPane.add(p);
		char ch = (char) 169;
		JLabel copyrightLabel = new JLabel(ch
				+ " 2017, Carnegie Mellon University.  All Rights Reserved.  ");
		copyrightLabel.setAlignmentX(Component.CENTER_ALIGNMENT);
		JPanel infopanel = new JPanel(new SpringLayout());
		infopanel.add(copyrightLabel);
		infopanel.add(infoButton);
		infoButton.addActionListener(this);
		infoButton.setBackground(DREM_IO.lightBlue);
		SpringUtilities.makeCompactGrid(infopanel, 1, 2, 0, 4, 0, 0);

		infopanel.setBackground(lightBlue);
		contentPane.add(infopanel);

		contentPane.add(Box.createRigidArea(new Dimension(0, 2)));

		makeOptionsDialog(); //create and show the option dialog

		theRepeatList = new ListDialog(this, DREM_IO.vRepeatFilesDEF,
				DREM_IO.balltimeDEF, repeatButton, DREM_IO.lightBlue,
				DREM_IO.defaultColor, DREM_IO.fc);
		miRNARepeatList = new ListDialog(this, DREM_IO.miRNARepeatFilesDEF,
				DREM_IO.miRNAalltimeDEF, miRNARepeatButton, DREM_IO.lightBlue,
				DREM_IO.defaultColor, DREM_IO.fc);
                ProteRepeatList = new ListDialog(this,DREM_IO.ProteRepeatFilesDEF,DREM_IO.ProtealltimeDEF,
                                ProteRepeatButton, DREM_IO.lightBlue, DREM_IO.defaultColor,DREM_IO.fc);
                               
	}

	/**
	 * Assigns the initial settings of the parameters based on the contents of
	 * szDefaultFile
	 */
	public static void parseDefaults() throws FileNotFoundException,
			IOException {
		String szLine;
		BufferedReader br;
		try {
			String szError = "";
			br = new BufferedReader(new FileReader(szDefaultFile));
			while ((szLine = br.readLine()) != null) {
				StringTokenizer st = new StringTokenizer(szLine, "\t");
				if (st.hasMoreTokens()) {
					String sztype = st.nextToken().trim();
					String szvalue = "";
					if (st.hasMoreTokens()) {
						szvalue = st.nextToken().trim();
					}

					if (!szvalue.equals("")) {
						if ((sztype
								.equalsIgnoreCase("Use_static_input_to_build_model"))
								|| (sztype
										.equalsIgnoreCase("Use_transcription_factor-gene_interaction_data_to_build"))
								|| (sztype
										.equalsIgnoreCase("Use_transcription_factor_gene_interaction_data_to_build"))) {
							if (szvalue.equalsIgnoreCase("true")) {
								bstaticsearchDEF = true;
							} else if (szvalue.equalsIgnoreCase("false")) {
								bstaticsearchDEF = false;
							} else {
								szError += "Warning: " + szvalue
										+ " is an unrecognized " + "value for "
										+ sztype + " "
										+ "(expecting true or false)";
							}
						} else if (sztype
								.equalsIgnoreCase("Static_Input_Data_File")
								|| sztype
										.equalsIgnoreCase("TF_gene_Interactions_File")
								|| sztype
										.equalsIgnoreCase("TF-gene_Interactions_File")) {
							szStaticFileDEF = szvalue;
						} else if (sztype.equalsIgnoreCase("Data_File")
								|| sztype
										.equalsIgnoreCase("Expression_Data_File")) {
							szDataFileDEF = szvalue;
						} else if (sztype
								.equalsIgnoreCase("Convergence_Likelihood_%")) {
							dCONVERGENCEDEF = Double.parseDouble(szvalue);
						} else if (sztype
								.equalsIgnoreCase("Minimum_Standard_Deviation")) {
							dMINSTDDEVALDEF = Double.parseDouble(szvalue);
						} else if (sztype.equalsIgnoreCase("Saved_Model_File")) {
							szInitFileDEF = szvalue;
						} else if ((sztype
								.equalsIgnoreCase("Spot_IDs_included_in_the_data_file"))
								|| (sztype
										.equalsIgnoreCase("Spot_IDs_included_in_the_the_data_file"))
								|| (sztype
										.equalsIgnoreCase("Spot_IDs_in_the_data_file"))) {
							bspotcheckDEF = (szvalue.equalsIgnoreCase("true"));
						} else if ((sztype.equalsIgnoreCase("Normalize_Data"))
								|| (sztype.equalsIgnoreCase("Transform_Data"))
								|| (sztype
										.equalsIgnoreCase("Transform_Data[Log transform data,Linear transform data,Add 0]"))
								|| (sztype
										.equalsIgnoreCase("Normalize_Data[Log normalize data,Normalize data,No normalization/add 0]"))) {
							try {
								nnormalizeDEF = Integer.parseInt(szvalue);
								if ((nnormalizeDEF < 0) || (nnormalizeDEF > 2)) {
									throw new IllegalArgumentException(
											szvalue
													+ " is an invalid argument for Normalize_Data");
								}
							} catch (NumberFormatException ex) {
								if (szvalue
										.equalsIgnoreCase("Log normalize data")) {
									nnormalizeDEF = 0;
								} else if (szvalue
										.equalsIgnoreCase("Normalize data")) {
									nnormalizeDEF = 1;
								} else if (szvalue
										.equalsIgnoreCase("No normalization/add 0")) {
									nnormalizeDEF = 2;
								} else {
									throw new IllegalArgumentException(
											szvalue
													+ " is an invalid argument for Normalize_Data");
								}
							}
						} else if ((sztype
								.equalsIgnoreCase("Change_should_be_based_on[Maximum-Minimum,Difference From 0]"))
								|| (sztype
										.equalsIgnoreCase("Change_should_be_based_on"))) {
							if (szvalue.equalsIgnoreCase("Maximum-Minimum")) {
								bmaxminDEF = true;
							} else if (szvalue
									.equalsIgnoreCase("Difference From 0")) {
								bmaxminDEF = false;
							} else {
								szError += szvalue
										+ " is an invalid value of "
										+ "Change_should_be_based_on[Maximum-Minimum,Difference From 0]\n";
							}
						} else if (sztype
								.equalsIgnoreCase("Gene_Annotation_Source")) {
							try {
								ndbDEF = Integer.parseInt(szvalue);
								if ((ndbDEF < 0)
										|| (ndbDEF >= GoAnnotations.organisms.length)) {
									ndbDEF = 0;
								}
							} catch (NumberFormatException ex) {
								boolean bfound = false;
								int nsource = 0;
								while ((nsource < GoAnnotations.organisms.length)
										&& (!bfound)) {
									if (GoAnnotations.organisms[nsource]
											.equalsIgnoreCase(szvalue)) {
										bfound = true;
										ndbDEF = nsource;
									} else {
										nsource++;
									}
								}

								if (!bfound) {
									szError += "Warning: "
											+ szvalue
											+ " is an unrecognized "
											+ "type for Gene Annotation Source\n";
								}
							}
						} else if (sztype.equalsIgnoreCase("Allow_Path_Merges")) {
							ballowmergeDEF = (szvalue.equalsIgnoreCase("true"));
						} else if (sztype
								.equalsIgnoreCase("TF-gene_Interaction_Source")) {
							int numitems = staticsourceArray.length;
							try {
								nstaticsourceDEF = Integer.parseInt(szvalue);
								if ((nstaticsourceDEF < 0)
										|| (nstaticsourceDEF >= numitems)) {
									nstaticsourceDEF = 0;
								}
							} catch (NumberFormatException ex) {
								boolean bfound = false;
								int nsource = 0;
								while ((nsource < numitems) && (!bfound)) {
									if (((String) staticsourceArray[nsource])
											.equalsIgnoreCase(szvalue)) {
										bfound = true;
										nstaticsourceDEF = nsource;
									} else {
										nsource++;
									}
								}

								if (!bfound) {
									szError += "Warning: "
											+ szvalue
											+ " is an unrecognized "
											+ "type for TF-gene_Interaction_Source";
								}
							}
						} else if (sztype
								.equalsIgnoreCase("Cross_Reference_Source")) {
							int numitems = GoAnnotations.defaultxrefs.length;
							try {
								nxrefDEF = Integer.parseInt(szvalue);
								if ((nxrefDEF < 0) || (nxrefDEF >= numitems)) {
									nxrefDEF = 0;
								}
							} catch (NumberFormatException ex) {
								boolean bfound = false;
								int nsource = 0;
								while ((nsource < numitems) && (!bfound)) {
									if (((String) GoAnnotations.defaultxrefs[nsource])
											.equalsIgnoreCase(szvalue)) {
										bfound = true;
										nxrefDEF = nsource;
									} else {
										nsource++;
									}
								}

								if (!bfound) {
									szError += "Warning: "
											+ szvalue
											+ " is an unrecognized "
											+ "type for a Cross_Reference_Source";
								}
							}
						} else if (sztype
								.equalsIgnoreCase("Gene_Annotation_File")) {
							szGeneAnnotationFileDEF = szvalue;
						} else if (sztype
								.equalsIgnoreCase("Cross_Reference_File")) {
							szCrossRefFileDEF = szvalue;
						} else if ((sztype
								.equalsIgnoreCase("Repeat_Data_Files(comma delimited list)"))
								|| (sztype
										.equalsIgnoreCase("Repeat_Data_Files"))) {
							vRepeatFilesDEF = new Vector<String>();
							StringTokenizer stRepeatList = new StringTokenizer(
									szvalue, ",");
							while (stRepeatList.hasMoreTokens()) {
								vRepeatFilesDEF.add(stRepeatList.nextToken());
							}
						} else if ((sztype
								.equalsIgnoreCase("Repeat_Data_is_from"))
								|| (sztype
										.equalsIgnoreCase("Repeat_Data_is_from[Different time periods,The same time period]"))) {
							if (szvalue
									.equalsIgnoreCase("Different time periods")) {
								balltimeDEF = true;
							} else if (szvalue
									.equalsIgnoreCase("The same time period")) {
								balltimeDEF = false;
							} else if (!szvalue.equals("")) {
								szError += "WARNING: '"
										+ szvalue
										+ "' is an invalid value for "
										+ "Repeat Data is from it must be either "
										+ "'Different time periods' or 'The same time period'\n";
							}
						} else if (sztype
								.equalsIgnoreCase("Y-axis_Scale_Factor")) {
							dYaxisDEF = Double.parseDouble(szvalue);
						} else if (sztype
								.equalsIgnoreCase("Scale_Node_Areas_By_The_Factor")) {
							dnodekDEF = Double.parseDouble(szvalue);
						} else if (sztype
								.equalsIgnoreCase("X-axis_Scale_Factor")) {
							dXaxisDEF = Double.parseDouble(szvalue);
						} else if ((sztype.equalsIgnoreCase("X-axis_scale"))
								|| (sztype
										.equalsIgnoreCase("X-axis_scale_should_be"))
								|| (sztype
										.equalsIgnoreCase("X-axis_scale[Uniform,Based on Real Time]"))
								|| (sztype
										.equalsIgnoreCase("X-axis_scale_should_be[Uniform,Based on Real Time]"))) {
							if (szvalue.equalsIgnoreCase("Uniform")) {
								brealXaxisDEF = false;
							} else if (szvalue
									.equalsIgnoreCase("Based on Real Time")) {
								brealXaxisDEF = true;
							} else if (!szvalue.equals("")) {
								szError += "WARNING: '"
										+ szvalue
										+ "' is an invalid value for "
										+ "X-axis_scale it must be either 'Uniform'"
										+ "or 'Based on Real Time'.\n";
							}
						} else if (sztype
								.equalsIgnoreCase("Key_Input_X_p-val_10^-X")) {
							dKeyInputXDEF = Double.parseDouble(szvalue);
						} else if ((sztype
								.equalsIgnoreCase("Key_Input_Significance_Based_On["
										+ "Path Significance Conditional on Split,Path Significance Overall,Split Significance]"))
								|| (sztype
										.equalsIgnoreCase("Key_Input_Significance_Based_On["
												+ "Split Significance,Path Significance Conditional on Split,Path Significance Overall]"))) {
							try {
								nKeyInputTypeDEF = Integer.parseInt(szvalue);
								if ((nKeyInputTypeDEF < 0)
										|| (nKeyInputTypeDEF > 2)) {
									throw new IllegalArgumentException(
											szvalue
													+ " is an invalid argument for Key Input Significance Based On");
								} else {
									// so code maps to input order
									if (nKeyInputTypeDEF == 0)
										nKeyInputTypeDEF = 1;
									else if (nKeyInputTypeDEF == 1)
										nKeyInputTypeDEF = 2;
									else if (nKeyInputTypeDEF == 2)
										nKeyInputTypeDEF = 0;

								}
							} catch (NumberFormatException ex) {
								if (szvalue
										.equalsIgnoreCase("Split Significance")) {
									nKeyInputTypeDEF = 0;
								} else if (szvalue
										.equalsIgnoreCase("Path Significance Conditional on Split")) {
									nKeyInputTypeDEF = 1;
								} else if (szvalue
										.equalsIgnoreCase("Path Significance Overall")) {
									nKeyInputTypeDEF = 2;
								} else {
									throw new IllegalArgumentException(
											szvalue
													+ " is an invalid argument for Key_Input_Significance_Based_On");
								}
							}
						} else if (sztype
								.equalsIgnoreCase("Maximum_number_of_paths_out_of_split")) {
							numchildDEF = Integer.parseInt(szvalue);
						} else if ((sztype.equalsIgnoreCase("Split_Seed"))
								|| (sztype.equalsIgnoreCase("Random_Seed"))) {
							nSEEDDEF = Integer.parseInt(szvalue);
						} else if (sztype
								.equalsIgnoreCase("Penalized_likelihood_node_penalty")) {
							dNODEPENALTYDEF = Double.parseDouble(szvalue);
						} else if ((sztype
								.equalsIgnoreCase("Model_selection_framework"))
								|| (sztype
										.equalsIgnoreCase("Model_selection_framework[Penalized Likelihood,Train-Test]"))) {
							try {
								int ntempval;
								ntempval = Integer.parseInt(szvalue);
								if ((ntempval < 0) || (ntempval > 1)) {
									bPENALIZEDDEF = (ninitsearchDEF == 0);
									throw new IllegalArgumentException(
											szvalue
													+ " is an invalid argument for Model_selection_framework");
								}
							} catch (NumberFormatException ex) {
								if (szvalue
										.equalsIgnoreCase("Penalized Likelihood")) {
									bPENALIZEDDEF = true;
								} else if (szvalue
										.equalsIgnoreCase("Train-Test")) {
									bPENALIZEDDEF = false;
								} else if (!szvalue.equals("")) {
									szError += "WARNING: '"
											+ szvalue
											+ "' is an invalid value for "
											+ "Model_selection_framework "
											+ "it must be either "
											+ "'Use As Is', 'Start Search From', or 'Do Not Use'\n";
								}
							}
						} else if ((sztype
								.equalsIgnoreCase("Delay_path_improvement"))
								|| (sztype
										.equalsIgnoreCase("Delay_split_score_%"))) {
							dDELAYPATHDEF = Double.parseDouble(szvalue);
							if (dDELAYPATHDEF < 0) {
								throw new IllegalArgumentException(szvalue
										+ " is an invalid value for " + sztype
										+ " must be >= 0");
							}
						} else if (sztype
								.equalsIgnoreCase("Merge_path_score_%")) {
							dDMERGEPATHDEF = Double.parseDouble(szvalue);
							if (dDMERGEPATHDEF < 0) {
								throw new IllegalArgumentException(szvalue
										+ " is an invalid value for " + sztype
										+ " must be >= 0");
							}
						} else if (sztype
								.equalsIgnoreCase("Merge_path_difference_threshold")) {
							dDMERGEPATHDIFFDEF = Double.parseDouble(szvalue);
							if (dDMERGEPATHDIFFDEF > 0) {
								throw new IllegalArgumentException(szvalue
										+ " is an invalid value for " + sztype
										+ " must be <= 0");
							}
						} else if (sztype
								.equalsIgnoreCase("Delay_split_difference_threshold")) {
							dDELAYPATHDIFFDEF = Double.parseDouble(szvalue);
							if (dDELAYPATHDIFFDEF > 0) {
								throw new IllegalArgumentException(szvalue
										+ " is an invalid value for " + sztype
										+ " must be <= 0");
							}
						} else if (sztype
								.equalsIgnoreCase("Delete_path_difference_threshold")) {
							dPRUNEPATHDIFFDEF = Double.parseDouble(szvalue);
							if (dPRUNEPATHDIFFDEF > 0) {
								throw new IllegalArgumentException(szvalue
										+ " is an invalid value for " + sztype
										+ " must be <= 0");
							}
						} else if (sztype
								.equalsIgnoreCase("Main_search_difference_threshold")) {
							dMinScoreDIFFDEF = Double.parseDouble(szvalue);
							if (dMinScoreDIFFDEF < 0) {
								throw new IllegalArgumentException(szvalue
										+ " is an invalid value for " + sztype
										+ " must be >= 0");
							}
						} else if ((sztype
								.equalsIgnoreCase("Prune_path_improvement"))
								|| (sztype
										.equalsIgnoreCase("Delete_path_score_%"))) {
							dPRUNEPATHDEF = Double.parseDouble(szvalue);
							if (dPRUNEPATHDEF < 0) {
								throw new IllegalArgumentException(szvalue
										+ " is an invalid value for " + sztype
										+ " must be >= 0");
							}
						} else if ((sztype
								.equalsIgnoreCase("Minimum_score_improvement"))
								|| (sztype
										.equalsIgnoreCase("Main_search_score_%"))) {
							dMinScoreDEF = Double.parseDouble(szvalue);
							if (dMinScoreDEF < 0) {
								throw new IllegalArgumentException(szvalue
										+ " is an invalid value for " + sztype
										+ " must be >= 0");
							}
						} else if ((sztype.equalsIgnoreCase("Saved_Model"))
								|| (sztype
										.equalsIgnoreCase("Saved_Model[Use As Is/Start Search From/Do Not Use]"))) {
							try {
								ninitsearchDEF = Integer.parseInt(szvalue);
								if ((ninitsearchDEF < 0)
										|| (ninitsearchDEF > 2)) {
									throw new IllegalArgumentException(
											szvalue
													+ " is an invalid argument for Saved_Model");
								}
							} catch (NumberFormatException ex) {
								if (szvalue.equalsIgnoreCase("Use As Is")) {
									ninitsearchDEF = 0;
								} else if (szvalue
										.equalsIgnoreCase("Start Search From")) {
									ninitsearchDEF = 1;
								} else if (szvalue
										.equalsIgnoreCase("Do Not Use")) {
									ninitsearchDEF = 2;
								} else if (!szvalue.equals("")) {
									szError += "WARNING: '"
											+ szvalue
											+ "' is an invalid value for "
											+ "Saved_Model "
											+ "it must be either "
											+ "'Use As Is', 'Start Search From', or 'Do Not Use'\n";
								}
							}
						} else if (sztype
								.equalsIgnoreCase("Filter_Gene_If_It_Has_No_Static_Input_Data")) {
							bfilterstaticDEF = (szvalue
									.equalsIgnoreCase("true"));
						} else if (sztype
								.equalsIgnoreCase("Maximum_Number_of_Missing_Values")) {
							nMaxMissingDEF = Integer.parseInt(szvalue);
						} else if (sztype
								.equalsIgnoreCase("Minimum_Absolute_Log_Ratio_Expression")) {
							dMinExpressionDEF = Double.parseDouble(szvalue);
						} else if (sztype
								.equalsIgnoreCase("Minimum_Correlation_between_Repeats")) {
							dMinCorrelationRepeatsDEF = Double
									.parseDouble(szvalue);
						} else if (sztype
								.equalsIgnoreCase("Pre-filtered_Gene_File")) {
							szPrefilteredDEF = szvalue;
						} else if (sztype
								.equalsIgnoreCase("Include_Biological_Process")) {
							bpontoDEF = (szvalue.equalsIgnoreCase("true"));
						} else if (sztype
								.equalsIgnoreCase("Include_Molecular_Function")) {
							bfontoDEF = (szvalue.equalsIgnoreCase("true"));
						} else if (sztype
								.equalsIgnoreCase("Include_Cellular_Process")) {
							bcontoDEF = (szvalue.equalsIgnoreCase("true"));
						} else if (sztype
								.equalsIgnoreCase("Only_include_annotations_with_these_evidence_codes")) {
							szevidenceDEF = szvalue;
						} else if (sztype
								.equalsIgnoreCase("Only_include_annotations_with_these_taxon_IDs")) {
							sztaxonDEF = szvalue;
						} else if ((sztype.equalsIgnoreCase("Category_ID_File"))
								|| (sztype
										.equalsIgnoreCase("Category_ID_Mapping_File"))) {
							szcategoryIDDEF = szvalue;
						} else if ((sztype
								.equalsIgnoreCase("GO_Minimum_number_of_genes"))
								|| (sztype
										.equalsIgnoreCase("Minimum_number_of_genes"))) {
							nMinGoGenesDEF = Integer.parseInt(szvalue);
						} else if (sztype.equalsIgnoreCase("Minimum_GO_level")) {
							nMinGOLevelDEF = Integer.parseInt(szvalue);
						} else if (sztype
								.equalsIgnoreCase("Minimum_Split_Percent")) {
							dpercentDEF = Double.parseDouble(szvalue);
						} else if (sztype
								.equalsIgnoreCase("Number_of_samples_for_randomized_multiple_hypothesis_correction")) {
							nSamplesMultipleDEF = Integer.parseInt(szvalue);
						} else if ((sztype
								.equalsIgnoreCase("Multiple_hypothesis_correction_method_enrichment[Bonferroni,Randomization]"))
								|| (sztype
										.equalsIgnoreCase("Multiple_hypothesis_correction_method[Bonferroni,Randomization]"))) {
							if (szvalue.equalsIgnoreCase("Bonferroni")) {
								brandomgoDEF = false;
							} else if (szvalue
									.equalsIgnoreCase("Randomization")) {
								brandomgoDEF = true;
							} else if (!szvalue.equals("")) {
								szError += "WARNING: '"
										+ szvalue
										+ "' is an invalid value for "
										+ "Correction_Method it must be either 'Bonferroni'"
										+ "or 'Randomization'.\n";
							}
						}
						else if (sztype.equalsIgnoreCase("miRNA-gene_Interaction_Source")) {
							miRNAInteractionDataFile = szvalue;
						} else if (sztype.equalsIgnoreCase("miRNA_Expression_Data_File")) {
							miRNAExpressionDataFile = szvalue;
						} else if (sztype.equalsIgnoreCase("Regulator_Types_Used_For_Activity_Scoring")) {
							if(szvalue.equalsIgnoreCase("None")){
								checkStatusTF = false;
								checkStatusmiRNA = false;
							} else if(szvalue.equalsIgnoreCase("TF")){
								checkStatusTF = true;
								checkStatusmiRNA = false;
							} else if (szvalue.equalsIgnoreCase("miRNA")) {
								checkStatusTF = false;
								checkStatusmiRNA = true;
							} else if (szvalue.equalsIgnoreCase("Both")) {
								checkStatusTF = true;
								checkStatusmiRNA = true;
							} else if (!szvalue.equals("")){
								szError = "WARNING: '"
									+ szvalue
									+ "' is an invalid value for "
									+ "Regulator_Types_Used_For_Activity_Scoring it must be either 'None', 'TF'"
									+ ", 'miRNA' or 'Both'.\n";
							}
						} else if (sztype.equalsIgnoreCase("Normalize_miRNA_Data[Log normalize data,Normalize data,No normalization/add 0]")) {
							if (szvalue.equalsIgnoreCase("Log normalize data")) {
								miRNATakeLog = true;
								miRNAAddZero = false;
							} else if (szvalue.equalsIgnoreCase("Normalize data")) {
								miRNATakeLog = false;
								miRNAAddZero = false;
							} else if (szvalue.equalsIgnoreCase("No normalization/add 0")) {
								miRNATakeLog = false;
								miRNAAddZero = true;
							} else {
								throw new IllegalArgumentException(
										szvalue+ " is an invalid argument for Normalize_miRNA_Data");
							}
						} else if ((sztype
								.equalsIgnoreCase("Repeat_miRNA_Data_Files(comma delimited list)"))
								|| (sztype
										.equalsIgnoreCase("Repeat_miRNA_Data_Files"))) {
							miRNARepeatFilesDEF = new Vector();
							StringTokenizer stRepeatList = new StringTokenizer(
									szvalue, ",");
							while (stRepeatList.hasMoreTokens()) {
								miRNARepeatFilesDEF.add(stRepeatList.nextToken());
							}
						} else if ((sztype.equalsIgnoreCase("Repeat_miRNA_Data_is_from"))
								|| (sztype
										.equalsIgnoreCase("Repeat_miRNA_Data_is_from[Different time periods,The same time period]"))) {
							if (szvalue.equalsIgnoreCase("Different time periods")) {
								miRNAalltimeDEF = true;
							} else if (szvalue.equalsIgnoreCase("The same time period")) {
								miRNAalltimeDEF = false;
							} else if (!szvalue.equals("")) {
								szError += "WARNING: '"
										+ szvalue
										+ "' is an invalid value for "
										+ "Repeat miRNA Data is from it must be either "
										+ "'Different time periods' or 'The same time period'\n";
							}
						} else if (sztype.equalsIgnoreCase("Filter_miRNA_With_No_Expression_Data_From_Regulators")) {
							filtermiRNAExp = (szvalue.equalsIgnoreCase("true"));
						} else if (sztype.equalsIgnoreCase("Expression_Scaling_Weight")) {
							miRNAWeight = Double.parseDouble(szvalue);
						} else if (sztype.equalsIgnoreCase("Minimum_TF_Expression_After_Scaling")) {
							tfWeight = Double.parseDouble(szvalue);
						} else if (sztype.equalsIgnoreCase("Regulator_Score_File")) {
							regScoreFile = szvalue;
						} else if (sztype.equalsIgnoreCase("DECOD_Executable_Path")) {
							decodPath = szvalue;
						} else if (sztype.equalsIgnoreCase("Gene_To_Fasta_Format_file")) {
							fastaFile = szvalue;
						} else if (sztype.equalsIgnoreCase("Active_TF_influence")) {
							dProbBindingFunctional = Double
									.parseDouble(szvalue);
							System.out
									.println("Setting active TF influence to "
											+ dProbBindingFunctional);
						} else if ((sztype.charAt(0) != '#')) {
							szError += "WARNING: '" + sztype
									+ "' is an unrecognized variable.\n";
						}
					}
				}
			}
			br.close();
			if (!szError.equals("")) {
				throw new IllegalArgumentException(szError);
			}
		} catch (FileNotFoundException ex) {
		}
	}

	/**
	 * Checks if the two data sets have the same number of rows, time points,
	 * and the gene name matches.
	 */
	public static void errorcheck(DREM_DataSet theDataSet1,
			DREM_DataSet theOtherSet) {

		if (theDataSet1.numcols != theOtherSet.numcols) {
			throw new IllegalArgumentException(
					"Repeat data set must have same "
							+ "number of columns as original, expecting "
							+ theDataSet1.numcols + " found "
							+ theOtherSet.numcols + " in the repeat");
		} else if (theDataSet1.numrows != theOtherSet.numrows) {
			throw new IllegalArgumentException(
					"Repeat data set must have same "
							+ "number of spots as the original, expecting "
							+ theDataSet1.numrows + " found "
							+ theOtherSet.numrows + " in the repeat");
		} else {
			for (int nrow = 0; nrow < theDataSet1.numrows; nrow++) {
				if (!theDataSet1.genenames[nrow]
						.equals(theOtherSet.genenames[nrow])) {
					throw new IllegalArgumentException("In row " + nrow
							+ " of the repeat set " + "expecting gene symbol "
							+ theDataSet1.genenames[nrow] + " found "
							+ theOtherSet.genenames[nrow]);
				} else if (!theDataSet1.probenames[nrow]
						.equals(theOtherSet.probenames[nrow])) {
					throw new IllegalArgumentException("In row " + nrow
							+ " of the repeat set " + "expecting gene symbol "
							+ theDataSet1.probenames[nrow] + " found "
							+ theOtherSet.probenames[nrow]);
				}
			}
		}
	}

	/**
	 * Checks if origcols and nrepeat cols are the same value, the length of
	 * origgenes and repeatgenes is the same, and the gene names are the same
	 */
	public static void errorcheck(String[] origgenes, String[] repeatgenes,
			int norigcols, int nrepeatcols) {
		if (norigcols != nrepeatcols) {
			throw new IllegalArgumentException(
					"Repeat data set must have same "
							+ "number of columns as original, expecting "
							+ norigcols + " found " + nrepeatcols
							+ " in the repeat");
		} else if (origgenes.length != repeatgenes.length) {
			throw new IllegalArgumentException(
					"Repeat data set must have same "
							+ "number of spots as the original, expecting "
							+ origgenes.length + " found " + repeatgenes.length
							+ " in the repeat");
		} else {
			for (int nrow = 0; nrow < origgenes.length; nrow++) {
				if (!origgenes[nrow].equals(repeatgenes[nrow])) {
					throw new IllegalArgumentException("In row " + nrow
							+ " of the repeat set " + "expecting gene symbol "
							+ origgenes[nrow] + " found " + repeatgenes[nrow]);
				} else if (!origgenes[nrow].equals(repeatgenes[nrow])) {
					throw new IllegalArgumentException("In row " + nrow
							+ " of the repeat set " + "expecting gene symbol "
							+ origgenes[nrow] + " found " + repeatgenes[nrow]);
				}
			}
		}
	}

	/**
	 * Returns a DREM_DataSet based on the provided input parameters
	 */
	synchronized public static DREM_DataSet buildset(
			String szorganismsourceval, String szxrefsourceval,
			String szxrefval, String szexp1val, String szgoval,
			String szgocategoryval, int nmaxmissing, double dexpressedval,
			double dmincorrelation, int nsamplespval, int nmingo,
			int nmingolevel, String szextraval, boolean balltime,
			Vector<String> repeatnames, boolean btakelog,
			boolean bspotincluded, boolean badd0, String szcategoryIDval,
			String szevidenceval, String sztaxonval, boolean bpontoval,
			boolean bcontoval, boolean bfontoval, boolean brandomgoval,
			boolean bmaxminval) throws Exception {
		DREM_DataSet theDataSetsMerged = null;
		if (balltime) {
			DREM_DataSet theDataSet1 = new DREM_DataSet(szexp1val, nmaxmissing,
					dexpressedval, dmincorrelation, btakelog, bspotincluded,
					false, badd0, bmaxminval, balltime);

			if (theDataSet1.numcols <= 1) {
				theDataSet1 = new DREM_DataSet(theDataSet1.filterDuplicates(),
						new DREM_GoAnnotations(szorganismsourceval,
								szxrefsourceval, szxrefval, szgoval,
								szgocategoryval, theDataSet1.genenames,
								theDataSet1.probenames, nsamplespval, nmingo,
								nmingolevel, szextraval, szcategoryIDval,
								bspotincluded, szevidenceval, sztaxonval,
								bpontoval, bcontoval, bfontoval, brandomgoval));

				DREM_DataSet theDataSet1fm;
				if (theDataSet1.numcols == 1) {
					theDataSet1fm = new DREM_DataSet(theDataSet1
							.filterMissing1point(), theDataSet1.tga);
					theDataSet1fm = new DREM_DataSet(theDataSet1fm
							.filtergenesthreshold1point(), theDataSet1fm.tga);
				} else {
					theDataSet1fm = theDataSet1;
				}
				return theDataSet1fm;
			} else {
				String[] origgenes = theDataSet1.genenames;
				theDataSet1 = new DREM_DataSet(theDataSet1.logratio2(),
						theDataSet1.tga);
				theDataSet1 = new DREM_DataSet(theDataSet1
						.averageAndFilterDuplicates(), theDataSet1.tga);

				// genevalues in log ratio before averaging stored
				// need for each gene duplicated
				// a mutlidimensional array of time series for each occurence

				int numrepeats = repeatnames.size();

				if (numrepeats > 0) {
					DREM_DataSet[] repeatSets = new DREM_DataSet[numrepeats];
					for (int nset = 0; nset < numrepeats; nset++) {
						String szfile = (String) repeatnames.get(nset);

						DREM_DataSet theOtherSet = new DREM_DataSet(szfile,
								nmaxmissing, dexpressedval, dmincorrelation,
								btakelog, bspotincluded, true, badd0,
								bmaxminval, balltime);
						errorcheck(origgenes, theOtherSet.genenames,
								theDataSet1.numcols, theOtherSet.numcols);
						// compute log ratio of each time series first then
						// merge
						// normalize the data
						theOtherSet = new DREM_DataSet(theOtherSet.logratio2(),
								theOtherSet.tga);
						theOtherSet = new DREM_DataSet(theOtherSet
								.averageAndFilterDuplicates(), theOtherSet.tga);
						// gene values in log ratio before averaging stored

						repeatSets[nset] = theOtherSet;
					}
					theDataSetsMerged = new DREM_DataSet(theDataSet1
							.mergeDataSets(repeatSets), theDataSet1.tga);
					theDataSetsMerged = new DREM_DataSet(theDataSetsMerged
							.filterdistprofiles(theDataSet1, repeatSets),
							theDataSetsMerged.tga);
				} else {
					theDataSetsMerged = theDataSet1;
				}

				theDataSetsMerged = new DREM_DataSet(theDataSetsMerged
						.filterMissing(), theDataSetsMerged.tga);
				theDataSetsMerged = new DREM_DataSet(theDataSetsMerged
						.filtergenesthreshold2(), theDataSetsMerged.tga);

				theDataSetsMerged.tga = new DREM_GoAnnotations(
						szorganismsourceval, szxrefsourceval, szxrefval,
						szgoval, szgocategoryval, theDataSet1.genenames,
						theDataSet1.probenames, nsamplespval, nmingo,
						nmingolevel, szextraval, szcategoryIDval,
						bspotincluded, szevidenceval, sztaxonval, bpontoval,
						bcontoval, bfontoval, brandomgoval);

				theDataSetsMerged.addExtraToFilter(theDataSetsMerged.tga);
				return theDataSetsMerged;
			}
		} else {
			DREM_DataSet theDataSet1 = new DREM_DataSet(szexp1val, nmaxmissing,
					dexpressedval, dmincorrelation, btakelog, bspotincluded,
					false, badd0, bmaxminval, balltime);
			if (theDataSet1.numcols <= 1) {
				theDataSet1 = new DREM_DataSet(theDataSet1.filterDuplicates(),
						new DREM_GoAnnotations(szorganismsourceval,
								szxrefsourceval, szxrefval, szgoval,
								szgocategoryval, theDataSet1.genenames,
								theDataSet1.probenames, nsamplespval, nmingo,
								nmingolevel, szextraval, szcategoryIDval,
								bspotincluded, szevidenceval, sztaxonval,
								bpontoval, bcontoval, bfontoval, brandomgoval));
				DREM_DataSet theDataSet1fm;
				if (theDataSet1.numcols == 1) {
					theDataSet1fm = new DREM_DataSet(theDataSet1
							.filterMissing1point(), theDataSet1.tga);
					theDataSet1fm = new DREM_DataSet(theDataSet1fm
							.filtergenesthreshold1point(), theDataSet1fm.tga);
				} else {
					theDataSet1fm = theDataSet1;
				} 
				return theDataSet1fm;
			} else {
				int numrepeats = repeatnames.size();

				if (numrepeats > 0) {
					DREM_DataSet[] repeatSets = new DREM_DataSet[numrepeats];
					for (int nset = 0; nset < numrepeats; nset++) {
						String szfile = (String) repeatnames.get(nset);

						DREM_DataSet theOtherSet = new DREM_DataSet(szfile,
								nmaxmissing, dexpressedval, dmincorrelation,
								btakelog, bspotincluded, true, badd0,
								bmaxminval, balltime);

						errorcheck(theDataSet1, theOtherSet);

						repeatSets[nset] = theOtherSet;
					}
					theDataSetsMerged = new DREM_DataSet(theDataSet1
							.mergeDataSets(repeatSets), theDataSet1.tga);
				} else {
					theDataSetsMerged = theDataSet1;
				}

				theDataSetsMerged = new DREM_DataSet(theDataSetsMerged
						.logratio2(), theDataSetsMerged.tga);
				theDataSetsMerged = new DREM_DataSet(theDataSetsMerged
						.averageAndFilterDuplicates(), theDataSetsMerged.tga);
				// gene values before averaging stored
				theDataSetsMerged = new DREM_DataSet(theDataSetsMerged
						.filterMissing(), theDataSetsMerged.tga);
				theDataSetsMerged = new DREM_DataSet(theDataSetsMerged
						.filtergenesthreshold2(), theDataSetsMerged.tga);

				theDataSetsMerged.tga = new DREM_GoAnnotations(
						szorganismsourceval, szxrefsourceval, szxrefval,
						szgoval, szgocategoryval, theDataSet1.genenames,
						theDataSet1.probenames, nsamplespval, nmingo,
						nmingolevel, szextraval, szcategoryIDval,
						bspotincluded, szevidenceval, sztaxonval, bpontoval,
						bcontoval, bfontoval, brandomgoval);
			}
			theDataSetsMerged.addExtraToFilter(theDataSetsMerged.tga);

			return theDataSetsMerged;
		}
	}

	/**
	 * Returns a DREM_DataSet based on the provided input parameters for
	 * miRNA.
         * 
         *Create a miRNA dataset based on miRNA and Repeats (if any)
         * 
	 */
        
	synchronized public static DataSetCore buildMIRNAset(
			String szorganismsourceval, String szxrefsourceval,
			String szxrefval, String szexp1val, String szgoval,
			String szgocategoryval, int nmaxmissing, double dexpressedval,
			double dmincorrelation, int nsamplespval, int nmingo,
			int nmingolevel, String szextraval, boolean balltime,
			Vector<String> repeatnames, boolean btakelog,
			boolean bspotincluded, boolean badd0, String szcategoryIDval,
			String szevidenceval, String sztaxonval, boolean bpontoval,
			boolean bcontoval, boolean bfontoval, boolean brandomgoval,
			boolean bmaxminval) throws Exception {
		DataSetCore theDataSetCoresMerged = null;

		DREM_DataSet theDataSet1 = new DREM_DataSet(szexp1val, nmaxmissing,
				dexpressedval, dmincorrelation, btakelog, bspotincluded,
				false, badd0, bmaxminval, balltime);
		DataSetCore theDataSet1Core = (DataSetCore) theDataSet1;
		if (theDataSet1.numcols <= 1) {
			theDataSet1Core = theDataSet1Core.filterDuplicates();

			DataSetCore theDataSet1fmCore;
			if (theDataSet1Core.numcols == 1) {
				theDataSet1fmCore = theDataSet1Core.filterMissing1point();
				theDataSet1fmCore = theDataSet1fmCore.filtergenesthreshold1point();
			} else {
				theDataSet1fmCore = theDataSet1Core;
			}
			return theDataSet1fmCore;
		}
		
		if (balltime) {
			String[] origgenes = theDataSet1.genenames;
			theDataSet1Core = theDataSet1Core.logratio2();
			theDataSet1Core = theDataSet1Core.averageAndFilterDuplicates();
			// genevalues in log ratio before averaging stored
			// need for each gene duplicated
			// a mutlidimensional array of time series for each occurence
			int numrepeats = repeatnames.size();

			if (numrepeats > 0) {
				DataSetCore[] repeatSets = new DataSetCore[numrepeats];
				for (int nset = 0; nset < numrepeats; nset++) {
					String szfile = (String) repeatnames.get(nset);
					DREM_DataSet theOtherSet = new DREM_DataSet(szfile,
							nmaxmissing, dexpressedval, dmincorrelation,
							btakelog, bspotincluded, true, badd0,
							bmaxminval, balltime);
					errorcheck(origgenes, theOtherSet.genenames,
							theDataSet1.numcols, theOtherSet.numcols);
					// compute log ratio of each time series first then
					// merge
					// normalize the data
					DataSetCore theOtherSetCore = theOtherSet.logratio2();
					theOtherSetCore = theOtherSetCore.averageAndFilterDuplicates();
					// gene values in log ratio before averaging stored
					repeatSets[nset] = theOtherSetCore;
				}
				theDataSetCoresMerged = theDataSet1Core.mergeDataSets(repeatSets);
				theDataSetCoresMerged = theDataSetCoresMerged.filterdistprofiles(theDataSet1, repeatSets);
			} else {
				theDataSetCoresMerged = theDataSet1Core;
			}
			return theDataSetCoresMerged;
		} else {
			int numrepeats = repeatnames.size();

			if (numrepeats > 0) {
				DataSetCore[] repeatSets = new DataSetCore[numrepeats];
				for (int nset = 0; nset < numrepeats; nset++) {
					String szfile = (String) repeatnames.get(nset);
						DREM_DataSet theOtherSet = new DREM_DataSet(szfile,
							nmaxmissing, dexpressedval, dmincorrelation,
							btakelog, bspotincluded, true, badd0,
							bmaxminval, balltime);
					errorcheck(theDataSet1, theOtherSet);
					repeatSets[nset] = (DataSetCore)theOtherSet;
				}
				theDataSetCoresMerged = theDataSet1.mergeDataSets(repeatSets);
			} else {
				theDataSetCoresMerged = theDataSet1Core;
			}

			theDataSetCoresMerged = theDataSetCoresMerged.logratio2();
			theDataSetCoresMerged = theDataSetCoresMerged.averageAndFilterDuplicates();
			return theDataSetCoresMerged;
		}
	}
           
        
        /**
	 * Returns a DREM_DataSet based on the provided input parameters for
	 * Proteomics
         * 
	 */
        synchronized public static DataSetCore buildProteset(
			String szorganismsourceval, String szxrefsourceval,
			String szxrefval, String szexp1val, String szgoval,
			String szgocategoryval, int nmaxmissing, double dexpressedval,
			double dmincorrelation, int nsamplespval, int nmingo,
			int nmingolevel, String szextraval, boolean balltime,
			Vector<String> repeatnames, boolean btakelog,
			boolean bspotincluded, boolean badd0, String szcategoryIDval,
			String szevidenceval, String sztaxonval, boolean bpontoval,
			boolean bcontoval, boolean bfontoval, boolean brandomgoval,
			boolean bmaxminval) throws Exception {
            
                DataSetCore theDataSetCoresMerged = null;
		DREM_DataSet theDataSet1 = new DREM_DataSet(szexp1val, nmaxmissing,
				dexpressedval, dmincorrelation, btakelog, bspotincluded,
				false, badd0, bmaxminval, balltime);
		DataSetCore theDataSet1Core = (DataSetCore) theDataSet1;
		if (theDataSet1.numcols <= 1) {
			theDataSet1Core = theDataSet1Core.filterDuplicates();

			DataSetCore theDataSet1fmCore;
			if (theDataSet1Core.numcols == 1) {
				theDataSet1fmCore = theDataSet1Core.filterMissing1point();
				theDataSet1fmCore = theDataSet1fmCore.filtergenesthreshold1point();
			} else {
				theDataSet1fmCore = theDataSet1Core;
			}
			return theDataSet1fmCore;
		}
		
		if (balltime) {
			String[] origgenes = theDataSet1.genenames;
			theDataSet1Core = theDataSet1Core.logratio2();
			theDataSet1Core = theDataSet1Core.averageAndFilterDuplicates();
			// genevalues in log ratio before averaging stored
			// need for each gene duplicated
			// a mutlidimensional array of time series for each occurence
			int numrepeats = repeatnames.size();

			if (numrepeats > 0) {
				DataSetCore[] repeatSets = new DataSetCore[numrepeats];
				for (int nset = 0; nset < numrepeats; nset++) {
					String szfile = (String) repeatnames.get(nset);
					DREM_DataSet theOtherSet = new DREM_DataSet(szfile,
							nmaxmissing, dexpressedval, dmincorrelation,
							btakelog, bspotincluded, true, badd0,
							bmaxminval, balltime);
					errorcheck(origgenes, theOtherSet.genenames,
							theDataSet1.numcols, theOtherSet.numcols);
					// compute log ratio of each time series first then
					// merge
					// normalize the data
					DataSetCore theOtherSetCore = theOtherSet.logratio2();
					theOtherSetCore = theOtherSetCore.averageAndFilterDuplicates();
					// gene values in log ratio before averaging stored
					repeatSets[nset] = theOtherSetCore;
				}
				theDataSetCoresMerged = theDataSet1Core.mergeDataSets(repeatSets);
				theDataSetCoresMerged = theDataSetCoresMerged.filterdistprofiles(theDataSet1, repeatSets);
			} else {
				theDataSetCoresMerged = theDataSet1Core;
			}
			return theDataSetCoresMerged;
		} else {
			int numrepeats = repeatnames.size();

			if (numrepeats > 0) {
				DataSetCore[] repeatSets = new DataSetCore[numrepeats];
				for (int nset = 0; nset < numrepeats; nset++) {
					String szfile = (String) repeatnames.get(nset);
						DREM_DataSet theOtherSet = new DREM_DataSet(szfile,
							nmaxmissing, dexpressedval, dmincorrelation,
							btakelog, bspotincluded, true, badd0,
							bmaxminval, balltime);
					errorcheck(theDataSet1, theOtherSet);
					repeatSets[nset] = (DataSetCore)theOtherSet;
				}
				theDataSetCoresMerged = theDataSet1.mergeDataSets(repeatSets);
			} else {
				theDataSetCoresMerged = theDataSet1Core;
			}

			theDataSetCoresMerged = theDataSetCoresMerged.logratio2();
			theDataSetCoresMerged = theDataSetCoresMerged.averageAndFilterDuplicates();
			return theDataSetCoresMerged;
		}
        }
       
	// ///////////////////////////////////////////////////////////////////////
	/**
	 * A control method that handles the response for when the execute button on
	 * the interface is pressed including building the data set, running the
	 * DREM modeling procedure, and displaying the results
	 */
        //FIXME: executing response function 
        
	public void clusterscript(
			String szstaticFileval, // szstaticFieldval -- before reference
			// class variable
			String szxrefval, String szexp1val, String szgoval,
			String szgocategoryval, String szmaxmissingval,
			String szexpressedval, String szfilterthresholdval,
			String szsamplepval, String szmingoval, String szmingolevelval,
			String szextraval, boolean balltime, Vector<String> repeatnames,
			Vector<String> mirnarepeatnames, Vector<String> proterepeatnames, //proteomics repeat files
                        boolean btakelog,
			boolean bgetxref, boolean bgetgoann, boolean bspotincluded,
			boolean badd0, String szcategoryIDval, String szinitfileval,
			String szevidenceval, String sztaxonval, boolean bpontoval,
			boolean bcontoval, boolean bfontoval, boolean brandomgoval,
			boolean bmaxminval) throws Exception {

		synchronized (lockpd) {
			while (ntodownload > 0) {
				lockpd.wait();
			}
		}

		if (nexceptions == 0) {
			if (szstaticFileval.trim().equals("")) {
				throw new IllegalArgumentException(
						"No transcription factor gene interaction input file given!");
			}
			if ((!szstaticFileval.trim().equals(""))
					&& (!(new File(szstaticFileval)).exists())) {
				throw new IllegalArgumentException(
						"The transcription factor gene interaction input file '"
								+ szstaticFileval + "' cannot be found.");
			}
			if (szexp1val.trim().equals("")) {
				throw new IllegalArgumentException(
						"No time series input data file given!");
			} else if (!(new File(szexp1val)).exists()) {
				throw new IllegalArgumentException(
						"The time series input data file '" + szexp1val
								+ "' cannot be found.");
			}

			if (szinitfileval.trim().equals("")) {
				szinitfileval = "";
			} else if (!(new File(szinitfileval)).exists()) {
				throw new IllegalArgumentException("The initial model file '"
						+ szinitfileval + "' cannot be found.");
			}

			if (szcategoryIDval.trim().equals("")) {
				szcategoryIDval = "";
			} else if (!(new File(szcategoryIDval)).exists()) {
				throw new IllegalArgumentException("The category ID file '"
						+ szcategoryIDval + "' cannot be found.");
			}

			if (szxrefval.trim().equals("")) {
				szxrefval = "";
			} else if ((!bgetxref) && !(new File(szxrefval)).exists()) {
				throw new IllegalArgumentException("The cross reference file '"
						+ szxrefval + "' cannot be found.");
			}

			if (szgoval.trim().equals("")) {
				szgoval = "";
			} else if ((!bgetgoann) && (!(new File(szgoval)).exists())) {
				throw new IllegalArgumentException("The GO annotation file '"
						+ szgoval + "' cannot be found.");
			}

			if (szextraval.trim().equals("")) {
				szextraval = "";
			} else if (!(new File(szextraval)).exists()) {
				throw new IllegalArgumentException(
						"The pre-filtered gene list file '" + szextraval
								+ "' cannot be found.");
			}

			if (szgocategoryval.trim().equals("")) {
				szgocategoryval = "";
			}

			int nmaxmissing;
			try {
				nmaxmissing = Integer.parseInt(szmaxmissingval);
				if (nmaxmissing < 0) {
					throw new IllegalArgumentException(
							"Maximum missing values must be positive");
				}
			} catch (NumberFormatException ex) {
				throw new IllegalArgumentException(
						"Maximum missing values must be an integer");
			}

			for (int nrepeat = 0; nrepeat < repeatnames.size(); nrepeat++) {
				if (!(new File((String) repeatnames.get(nrepeat))).exists()) {
					throw new IllegalArgumentException("The repeat data file '"
							+ repeatnames.get(nrepeat) + "' cannot be found");
				}
			}

			for (int nrepeat = 0; nrepeat < mirnarepeatnames.size(); nrepeat++) {
				if (!(new File((String) mirnarepeatnames.get(nrepeat)))
						.exists()) {
					throw new IllegalArgumentException("The repeat data file '"
							+ mirnarepeatnames.get(nrepeat)
							+ "' cannot be found");
				}
			}

			double dmincorrelation = Double.parseDouble(szfilterthresholdval);
			if ((dmincorrelation < -1.1) || (dmincorrelation > 1.1)) {
				throw new IllegalArgumentException(
						"Correlation Lower Bound for Filtering must be in [-1.1,1.1]");
			}

			double dexpressedval = Double.parseDouble(szexpressedval);
			if (dexpressedval < -0.05) {
				throw new IllegalArgumentException(
						"Expression Value for filter must be >= -0.05");
			}

			int nmingo = Integer.parseInt(szmingoval);
			if (nmingo < 1) {
				throw new IllegalArgumentException(
						"Minimum number of GO genes must be at least 1");
			}

			int nmingolevel = Integer.parseInt(szmingolevelval);
			if (nmingolevel < 1) {
				throw new IllegalArgumentException(
						"Minimum number of GO level must be at least 1");
			}

			int nsamplespval;

			try {
				nsamplespval = Integer.parseInt(szsamplepval);
			} catch (NumberFormatException ex) {
				throw new IllegalArgumentException(
						"Number of samples for p-value correction must be an integer");
			}

			if (nsamplespval < 1) {
				throw new IllegalArgumentException(
						"Number of samples for p-value correction must be positive");
			}

			executeDialognf = new JDialog(this, "Executing...");

			executeDialognf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
			final JDialog executeDialog = executeDialognf;
			final JFrame thisframe = this;

			statusLabel = new JLabel(" Starting Search... ");
			statuscountLabel = new JLabel(" ");
			statusLabel15 = new JLabel(" ");
			statusLabel2 = new JLabel(" ");
			statusLabel3 = new JLabel(" ");

			bendsearch = false;
			if ((ninitsearchval != 0) || (szinitfileval.equals(""))) {
				javax.swing.SwingUtilities.invokeAndWait(new Runnable() {
					public void run() {
						Container cp = executeDialog.getContentPane();
						JPanel lp = new JPanel(new SpringLayout());
						lp.setBackground(Color.white);
						lp
								.add(new JLabel(
										" Please wait this may take several minutes..."
												+ "                                  "));
						JPanel breakPanel = new JPanel();
						breakPanel.setBackground(lightBlue);
						lp.add(breakPanel);
						lp.add(statuscountLabel);
						lp.add(statusLabel);
						lp.add(statusLabel15);
						JPanel breakPanel2 = new JPanel();
						breakPanel2.setBackground(lightBlue);
						lp.add(breakPanel2);
						lp.add(statusLabel3);
						lp.add(statusLabel2);

						JPanel breakPanel25 = new JPanel();
						breakPanel25.setBackground(lightBlue);
						lp.add(breakPanel25);

						currentButton = new JButton("Display Current Model");
						currentButton
								.addActionListener((ActionListener) thisframe);
						lp.add(currentButton);

						JPanel breakPanel3 = new JPanel();
						breakPanel3.setBackground(lightBlue);
						lp.add(breakPanel3);
						endSearchButton = new JButton("End Search");
						endSearchButton
								.addActionListener((ActionListener) thisframe);

						lp.add(endSearchButton);
						SpringUtilities
								.makeCompactGrid(lp, 12, 1, 0, 10, 0, 10);
						cp.add(lp);
						executeDialog.pack();
						executeDialog.setBackground(Color.white);
						cp.setBackground(Color.white);
						executeDialog.setLocation(thisframe.getX() + 250,
								thisframe.getY() + 200);
						executeDialog
								.setDefaultCloseOperation(DISPOSE_ON_CLOSE);
						executeDialog.setVisible(true);
					}
				});
			}

			final DREM_DataSet thefDataSetfmnel = DREM_IO.buildset(
					szorganismsourceval, szxrefsourceval, szxrefval, szexp1val,
					szgoval, szgocategoryval, nmaxmissing, dexpressedval,
					dmincorrelation, nsamplespval, nmingo, nmingolevel,
					szextraval, balltime, repeatnames, btakelog, bspotincluded,
					badd0, szcategoryIDval, szevidenceval, sztaxonval,
					bpontoval, bcontoval, bfontoval, brandomgoval, bmaxminval);

                        //FIXME miRNADataset
			DataSetCore theMIRNADataSet = null;
			if (checkStatusmiRNA && miRNAExpressionDataFile != null
					&& !miRNAExpressionDataFile.equals("")) {
                                                theMIRNADataSet = DREM_IO.buildMIRNAset(null, null, null,
						miRNAExpressionDataFile, "", null, nmaxmissing,
						dexpressedval, dmincorrelation, nsamplespval, nmingo,
						nmingolevel, szextraval, balltimemiRNA,
						mirnarepeatnames, miRNATakeLog, false, miRNAAddZero,
						szcategoryIDval, szevidenceval, sztaxonval, bpontoval,
						bcontoval, bfontoval, brandomgoval, bmaxminval);
			}
                        
                        //FIXME proteomics DataSet
                        //System.out.print(djProteFile);
                        //System.out.print(balltimeprote);
                        //System.out.print(proterepeatnames);
                        //System.out.print(proteTakeLog);
                        //System.out.print(proteAddZero);
                        //System.out.print(djMethyFile);
                        //System.out.print(djMethyGTF);
                        
                        DataSetCore theProteDataSet=null;
                        if((bProteAll||bProteTF) && !(djProteFile.equals(""))){ 
                            theProteDataSet=DREM_IO.buildProteset(null,null,null,
                                    djProteFile,"",null,nmaxmissing,
                                    dexpressedval,dmincorrelation,nsamplespval,nmingo,
                                    nmingolevel,szextraval,balltimeprote,
                                    proterepeatnames,proteTakeLog,false,proteAddZero,
                                    szcategoryIDval,szevidenceval,sztaxonval,bpontoval,
                                    bcontoval,bfontoval,brandomgoval,bmaxminval);
                         }
                        
                        // methylation data
                        HashMap<String,HashMap<String,Double>> methyGeneScoreMap=null;
                        
                        if (!(djMethyFile.equals("")) && !(djMethyGTF.equals(""))){
                             //update gene2RegBinding/gene2RegBindingIndex using methylation data
                            methyGeneScoreMap=getMethyScore(djMethyFile,djMethyGTF);
                        }
			// --------------------------------------------------------------------------------------------
                        
    			DREM_Timeiohmm thetimehmm = null;
                        
                        
                        //FIXME iohmm: pass all parameters here!!!!
			thetimehmm = new DREM_Timeiohmm(thefDataSetfmnel, szstaticFileval,
					sznumchildval, szepsilonval, szprunepathval,
					szdelaypathval, szmergepathval, szepsilonvaldiff,
					szprunepathvaldiff, szdelaypathvaldiff, szmergepathvaldiff,
					szseedval, bstaticcheckval, ballowmergeval, ninitsearchval,
					szinitfileval, statusLabel, statusLabel15, statusLabel2,
					statusLabel3, statuscountLabel, endSearchButton,
					bstaticsearchval, brealXaxisDEF, dYaxisDEF, dXaxisDEF,
					dnodekDEF, nKeyInputTypeDEF, dKeyInputXDEF, dpercentDEF,
					currentButton, "", sznodepenaltyval, bpenalizedmodelval,
					szconvergenceval, szminstddeval,
					staticsourceArray[nstaticsourcecb], checkStatusTF,
					checkStatusmiRNA, miRNAInteractionDataFile,
                                        szexp1val,
					theMIRNADataSet,miRNAExpressionDataFile, // miRNA data
                                        theProteDataSet,djProteFile,djPPIFile,bProteAll, // proteomics data
                                       
                                        methyGeneScoreMap,// methylation scores for gene promoters
                                        fastaFile, decodPath,
					regScoreFile, dProbBindingFunctional, miRNAWeight, tfWeight,
					filtermiRNAExp,proteweightval);

			((DREM_GoAnnotations) thetimehmm.theDataSet.tga).buildRecDREM(
					thetimehmm.treeptr, thetimehmm.theDataSet.genenames);

			thetimehmm.traverse(thetimehmm.treeptr, 0, true);
			thetimehmm.traverse(thetimehmm.treeptr, 0, false);

			final DREM_Timeiohmm fthetimehmm = thetimehmm;

			try {
				javax.swing.SwingUtilities.invokeAndWait(new Runnable() {
					public void run() {
						// not copying on last load
                                                // FIXME: final mode;
						final DREMGui theDREMGui = new DREMGui(fthetimehmm,
								fthetimehmm.treeptr, brealXaxisDEF, dYaxisDEF,
								dXaxisDEF, nKeyInputTypeDEF, dKeyInputXDEF,
								dpercentDEF, "(Final Model)", dnodekDEF);
						edu.umd.cs.piccolo.PCanvas.CURRENT_ZCANVAS = null;
						theDREMGui.setLocation(15, 40);
						theDREMGui
								.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

						theDREMGui.setVisible(true);
						theDREMGui.addWindowListener(new WindowAdapter() {
							public void windowClosing(WindowEvent we) {
								theDREMGui.closeWindows();
								if (!theDREMGui.bsavedchange) {
									Object[] options = { "Yes", "No" };
									int noption = JOptionPane
											.showOptionDialog(
													theDREMGui,
													"Would you like to save the model?",
													"Question",
													JOptionPane.YES_NO_OPTION,
													JOptionPane.QUESTION_MESSAGE,
													null, options, options[1]);

									if (noption == 0) {
										if (theDREMGui.saveModelFrame == null) {
											theDREMGui.saveModelFrame = new JFrame(
													"Save Model to File");
											theDREMGui.saveModelFrame
													.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
											theDREMGui.saveModelFrame
													.setLocation(400, 300);
											DREMGui_SaveModel newContentPane = new DREMGui_SaveModel(
													theDREMGui.theTimeiohmm,
													theDREMGui.theTimeiohmm.treeptr,
													theDREMGui.saveModelFrame,
													theDREMGui);
											newContentPane.setOpaque(true);
											// content panes must be opaque
											theDREMGui.saveModelFrame
													.setContentPane(newContentPane);
											// Display the window.
											theDREMGui.saveModelFrame.pack();
										} else {
											theDREMGui.saveModelFrame
													.setExtendedState(Frame.NORMAL);
										}
										theDREMGui.saveModelFrame
												.setVisible(true);
									}
								}
							}
						});

						// TODO tony's save file???
						if (bbatchMode && theDREMGui.theTimeiohmm.bindingData.regPriors != null) {
							theDREMGui.batchSave(saveFile);
							theDREMGui.dispose();
						}
					}
				});
			} catch (InterruptedException iex) {
			} catch (java.lang.reflect.InvocationTargetException itex) {
			}

			if (executeDialognf != null) {
				executeDialognf.dispose();
				executeDialognf.setVisible(false);
			}
		}
		long e1 = System.currentTimeMillis();
		System.out.println("Time: " + (e1 - s1) + "ms");

		if (bbatchMode) {
			this.dispose();
		}
	}

	// /////////////////////////////////////////////////////////////////////
	/**
	 * Updates the interface and internal values for the static TF-gene input
	 */
	public void handlestaticsource() {
		if (staticFileField.isEditable()) {
			szuserFileField = staticFileField.getText();
		}

		nstaticsourcecb = staticsourcecb.getSelectedIndex();

		if (nstaticsourcecb >= 1) {
			staticFileField.setText(SZSTATICDIR
					+ System.getProperty("file.separator")
					+ staticsourceArray[nstaticsourcecb]);
			staticFileField.setEditable(false);
			staticFileButton.setEnabled(false);
		} else {
			staticFileField.setText(szuserFileField);
			staticFileField.setEditable(true);
			staticFileButton.setEnabled(true);
		}
	}

	/**
	 * Updates the interface and internal values for the GO cross reference
	 * input
	 */
	public void handlenxrefval() {
		if (xrefcb.getSelectedIndex() >= 0) {
			if (xrefField.isEditable()) {
				szuserxref = xrefField.getText();
			}

			nxrefcb = xrefcb.getSelectedIndex();

			szxrefsourceval = GoAnnotations.defaultxrefs[nxrefcb];
			if (nxrefcb >= 2) {
				xrefField.setText(GoAnnotations.xreffile[nxrefcb]);
				xrefField.setEditable(false);
				xrefButton.setEnabled(false);
			} else if ((nxrefcb == 0) && (ndb != 1)) {
				xrefField.setText(szuserxref);
				xrefField.setEditable(true);
				xrefButton.setEnabled(true);
			} else {
				xrefField.setText("");
				xrefField.setEditable(false);
				xrefButton.setEnabled(false);
			}

			if ((nxrefcb >= 2) && (!GoAnnotations.xreffile[nxrefcb].equals(""))) {
				File xrefFile = new File(GoAnnotations.xreffile[nxrefcb]);
				if (xrefFile.exists()) {
					xrefcheck.setSelected(false);
					xrefcheck.setEnabled(true);
				} else {
					xrefcheck.setSelected(true);
					xrefcheck.setEnabled(false);
				}
			} else {
				xrefcheck.setSelected(false);
				xrefcheck.setEnabled(false);
			}
		}
	}

	/**
	 * Updates the interface and internal values for the GO cross reference
	 * input
	 */
	public void handlendbval() {
		if (xrefField.isEditable()) {
			szuserxref = xrefField.getText();
		}

		if (goField.isEditable()) {
			szusergann = goField.getText();
		}

		ndb = orgcb.getSelectedIndex();
		szorganismsourceval = GoAnnotations.organisms[ndb];

		int nitems = xrefcb.getItemCount();
		if (nitems == 1) {
			xrefcb.removeItemAt(0);
			for (int nindex = 0; nindex < GoAnnotations.defaultxrefs.length; nindex++) {
				xrefcb.addItem(GoAnnotations.defaultxrefs[nindex]);
			}
		}

		if (GoAnnotations.xreforgfile[ndb].equals("")) {
			if (ndb == 1) {
				xrefcb.removeAllItems();
				xrefcb.addItem(GoAnnotations.defaultxrefs[1]);
				// removing user provided option
			}
			xrefcb.setSelectedIndex(0);
		} else {
			xrefcb.setSelectedItem(GoAnnotations.organisms[ndb]);
		}

		if (ndb <= 1) {
			anncheck.setSelected(false);
			anncheck.setEnabled(false);
		} else {
			File goannFile = new File(GoAnnotations.gannfile[ndb]);
			if (goannFile.exists()) {
				anncheck.setSelected(false);
				anncheck.setEnabled(true);
			} else {
				anncheck.setSelected(true);
				anncheck.setEnabled(false);
			}
		}

		if (ndb == 0) {
			obocheck.setEnabled(true);
			obocheck.setSelected(false);
		} else if (ndb == 1) {
			obocheck.setSelected(false);
			obocheck.setEnabled(false);
		} else {
			File oboFile = new File(szgocategoryval);

			if (oboFile.exists()) {
				obocheck.setEnabled(true);
				obocheck.setSelected(false);
			} else {
				obocheck.setSelected(true);
				obocheck.setEnabled(false);
			}
		}

		handlenxrefval();
		if (ndb >= 1) {
			goField.setText(GoAnnotations.gannfile[ndb]);
			goField.setEditable(false);
			goLabelButton.setEnabled(false);
		} else {
			goLabelButton.setEnabled(true);
			goField.setText(szusergann);
			goField.setEditable(true);
		}
	}

	/**
	 * Responds to a change in the radio button selection specifying the model
	 * selection criteria
	 */
	public void stateChanged(ChangeEvent e) {
		Object esource = e.getSource();

		if (esource == traintestButton) {
			toggleEnabled(traintestButton.isSelected());
		}
                
                if (esource == ProteNoCheckBox){
                        toggleUseProteEnabled(!(ProteNoCheckBox.isSelected()));
                }
                
	}

	/**
	 * Responds to buttons being pressed on the main input interface
	 */
	public void actionPerformed(ActionEvent e) {
		Object esource = e.getSource();

		if (esource == endSearchButton) {
			bendsearch = true;
			statusLabel2
					.setText(" End Search Requested. Search Will End Soon. ");
			endSearchButton.setEnabled(false);
		} else if (esource == currentButton) {
			bdisplaycurrent = true;
			currentButton.setEnabled(false);
		} else if (esource == viewstaticButton) {
			final String szfile = staticFileField.getText();
			final JFrame fframe = this;
			if ((new File(szfile)).exists()) {
				javax.swing.SwingUtilities.invokeLater(new Runnable() {
					public void run() {
						DataTable newContentPane = new DataTable(fframe,
								szfile, true);
						JFrame dtframe = new JFrame(szfile);
						dtframe
								.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
						dtframe.setLocation(20, 50);
						newContentPane.setOpaque(true); // content panes must be
						// opaque
						dtframe.setContentPane(newContentPane);
						// Display the window.
						dtframe.pack();
						dtframe.setVisible(true);
					}
				});
			} else {
				javax.swing.SwingUtilities.invokeLater(new Runnable() {
					public void run() {
						JOptionPane.showMessageDialog(fframe, "File '" + szfile
								+ "' was not found.", "Error",
								JOptionPane.ERROR_MESSAGE);
					}
				});
			}
		} else if (esource == viewButton) {
			final String szfile = orig1Field.getText();
			final JFrame fframe = this;
			if ((new File(szfile)).exists()) {
				javax.swing.SwingUtilities.invokeLater(new Runnable() {
					public void run() {
						DataTable newContentPane = new DataTable(fframe,
								szfile, false);
						JFrame dtframe = new JFrame(szfile);
						dtframe
								.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
						dtframe.setLocation(20, 50);
						newContentPane.setOpaque(true); // content panes must be
						// opaque
						dtframe.setContentPane(newContentPane);
						// Display the window.
						dtframe.pack();
						dtframe.setVisible(true);
					}
				});
			} else {
				javax.swing.SwingUtilities.invokeLater(new Runnable() {
					public void run() {
						JOptionPane.showMessageDialog(fframe, "File '" + szfile
								+ "' was not found.", "Error",
								JOptionPane.ERROR_MESSAGE);
					}
				});
			}
		} else if (esource == staticsourcecb) {
			handlestaticsource();
		} else if (esource == orgcb) {
			handlendbval();
		} else if (esource == xrefcb) {
			handlenxrefval();
		} else if (esource == xrefButton) {
			int returnVal = fc.showOpenDialog(this);

			if (returnVal == JFileChooser.APPROVE_OPTION) {
				File file = fc.getSelectedFile();
				xrefField.setText(file.getAbsolutePath());
			}
		} else if (esource == orig1Button) {
			int returnVal = fc.showOpenDialog(this);

			if (returnVal == JFileChooser.APPROVE_OPTION) {
				File file = fc.getSelectedFile();
				orig1Field.setText(file.getAbsolutePath());
			}
		} else if (esource == staticFileButton) {
			int returnVal = fc.showOpenDialog(this);

			if (returnVal == JFileChooser.APPROVE_OPTION) {
				File file = fc.getSelectedFile();
				staticFileField.setText(file.getAbsolutePath());
			}
		} else if (esource == extraButton) {
			int returnVal = fc.showOpenDialog(this);

			if (returnVal == JFileChooser.APPROVE_OPTION) {
				File file = fc.getSelectedFile();
				extraField.setText(file.getAbsolutePath());
			}
		} else if (esource == initfileButton) {
			int returnVal = fc.showOpenDialog(this);

			if (returnVal == JFileChooser.APPROVE_OPTION) {
				File file = fc.getSelectedFile();
				initFileField.setText(file.getAbsolutePath());
			}
		} else if (esource == goLabelButton) {
			int returnVal = fc.showOpenDialog(this);

			if (returnVal == JFileChooser.APPROVE_OPTION) {
				File file = fc.getSelectedFile();
				goField.setText(file.getAbsolutePath());
			}
                } else if (esource ==MethyFileButton){
                        //FIXME: Methy response
                        int returnVal=fc.showOpenDialog(this);
                        if (returnVal==JFileChooser.APPROVE_OPTION){
                            File file=fc.getSelectedFile();
                            MethyField.setText(file.getAbsolutePath());
                        }
                } else if (esource ==MethyGTFButton){
                        int returnVal=fc.showOpenDialog(this);
                        if (returnVal==JFileChooser.APPROVE_OPTION){
                            File file=fc.getSelectedFile();
                            MethyGTFField.setText(file.getAbsolutePath());
                        }            
                } else if (esource== ProteFileButton){
                        int returnVal=fc.showOpenDialog(this);
                        if (returnVal==JFileChooser.APPROVE_OPTION){
                            File file =fc.getSelectedFile();
                            ProteField.setText(file.getAbsolutePath()); 
                        }
                } else if (esource==ProtePPIButton){
                        int returnVal=fc.showOpenDialog(this);
                        if (returnVal==JFileChooser.APPROVE_OPTION){
                            File file=fc.getSelectedFile();
                            ProtePPIField.setText(file.getAbsolutePath());
                        }
                } else if (esource==ProteRepeatButton){
                        //FIXME ProteRepeatButton
                        ProteRepeatList.setLocation(this.getX() + 75, this.getY() + 100);
                            
                        ProteRepeatList.setModal(true);
                        ProteRepeatList.setVisible(true);

		} else if (categoryIDButton == esource) {
			int returnVal = fc.showOpenDialog(this);

			if (returnVal == JFileChooser.APPROVE_OPTION) {
				File file = fc.getSelectedFile();
				categoryIDField.setText(file.getAbsolutePath());
			}
		} else if (esource == optionsButton) {
			theOptions.setLocation(this.getX() + 5, this.getY() + 100);
			theOptions.setVisible(true);
		} else if (esource == repeatButton) {
			theRepeatList.setLocation(this.getX() + 75, this.getY() + 100);
			theRepeatList.setVisible(true);
		} else if (esource == staticmiRNAInteractionFileButton) {
			int returnVal = fc.showOpenDialog(this);

			if (returnVal == JFileChooser.APPROVE_OPTION) {
				File file = fc.getSelectedFile();
				miRNAInteractionField.setText(file.getAbsolutePath());
			}
		} else if (esource == miRNACB) {
			if (miRNACB.getSelectedIndex() == 0) {
				miRNAInteractionField.setEditable(true);
				miRNAInteractionField.setEnabled(true);
				miRNAInteractionField.setText("");
				staticmiRNAInteractionFileButton.setEnabled(true);
				bgetmirnaexp = false;
			} else {
                                //FIXME: miRNAtarget
				miRNAInteractionField.setEditable(false);
				miRNAInteractionField.setEnabled(false);
                                String miRNAInput;  
                                final String miRNAInputDIR = "miRNAInput";
                                miRNAInput=miRNAInputDIR+"/"+miRNAFiles[miRNACB.getSelectedIndex()];
                                miRNAInteractionField.setText(miRNAInput);
                                
                                staticmiRNAInteractionFileButton.setEnabled(false);
				bgetmirnaexp = false; //disable the miRNA download
			}
		} else if (esource == miRNARepeatButton) {
			miRNARepeatList.setLocation(this.getX() + 75, this.getY() + 100);
			// setModal forces the miRNA repeat dialog to appear as the top
			// window
			// and blocks focus from other windows
			miRNARepeatList.setModal(true);
			miRNARepeatList.setVisible(true);
		} else if (esource == staticmiRNAExpressionFileButton) {
			int returnVal = fc.showOpenDialog(this);

			if (returnVal == JFileChooser.APPROVE_OPTION) {
				File file = fc.getSelectedFile();
				miRNAExpressionField.setText(file.getAbsolutePath());
			}
		} else if (esource == fastaDataFileButton) {
			int returnVal = fc.showOpenDialog(this);

			if (returnVal == JFileChooser.APPROVE_OPTION) {
				File file = fc.getSelectedFile();
				fastaDataField.setText(file.getAbsolutePath());
			}
		} else if (esource == decodPathButton) {
			int returnVal = fc.showOpenDialog(this);

			if (returnVal == JFileChooser.APPROVE_OPTION) {
				File file = fc.getSelectedFile();
				decodPathField.setText(file.getAbsolutePath());
			}
		} else if (esource == regScoreFileButton) {
			int returnVal = fc.showOpenDialog(this);

			if (returnVal == JFileChooser.APPROVE_OPTION) {
				File file = fc.getSelectedFile();
				regScoreField.setText(file.getAbsolutePath());
			}
		} else if (esource == clusterAButton) {
                        // action when execute is clicked
			s1 = System.currentTimeMillis();
			String szcommand = e.getActionCommand();

			szorig1val = orig1Field.getText();
			szstaticFileval = staticFileField.getText();
			szinitfileval = initFileField.getText();
			szxrefval = xrefField.getText();
			szorig2val = "";
			szgoval = goField.getText();
			szgocategoryval = szGeneOntologyFileDEF;
			szextraval = extraField.getText();
			szcategoryIDval = categoryIDField.getText();
			if (initopenButton.isSelected()) {
				ninitsearchval = 0;
			} else if (initsearchButton.isSelected()) {
				ninitsearchval = 1;
			} else {
				ninitsearchval = 2;
			}
                        //FIXME Proteomics inputs
                        djProteFile = ProteField.getText();
                        djPPIFile=ProtePPIField.getText();
                        bProteAll = ProteAllCheckBox.isSelected(); //use TF proteomics data only
                        bProteTF = ProteTFCheckBox.isSelected();  //use all proteomics data
                        proteTakeLog = ProtelognormButton.isSelected(); // determine the normalization method for proteomics data
			proteAddZero = ProtenonormButton.isSelected();
                        
                        //FIXME Methy inputs
                        djMethyFile=MethyField.getText();
                        djMethyGTF=MethyGTFField.getText();
                        
			szmaxmissingval = thespinnermaxmissing.getValue().toString();
			szexpressval = thespinnerexpress.getValue().toString();
			szfilterthresholdval = thespinnerfilterthreshold.getValue()
					.toString();
			szsamplepvalval = thespinnersamplepval.getValue().toString();
			szmingoval = thespinnermingo.getValue().toString();
			szepsilonval = thespinnerepsilon.getValue().toString();
			szprunepathval = thespinnerprunepath.getValue().toString();
			szdelaypathval = thespinnerdelaypath.getValue().toString();
			szmergepathval = thespinnermergepath.getValue().toString();
			sznodepenaltyval = thespinnernodepenalty.getValue().toString();
			szconvergenceval = thespinnerconvergence.getValue().toString();
			szminstddeval = thespinnerminstddev.getValue().toString();
                        
                        //relative proteomics weight values
                        
                        proteweightval=theproteWeight.getValue().toString();
                        //epiweightval=theproteWeight.getValue().toString();
                        
			szepsilonvaldiff = thespinnerepsilondiff.getValue().toString();
			szprunepathvaldiff = thespinnerprunepathdiff.getValue().toString();
			szdelaypathvaldiff = thespinnerdelaypathdiff.getValue().toString();
			szmergepathvaldiff = thespinnermergepathdiff.getValue().toString();

			szseedval = thespinnerseed.getValue().toString();
			szmingolevelval = thespinnermingolevel.getValue().toString();
			btakelog = lognormButton.isSelected();
			bspotincluded = spotcheck.isSelected();
			sznumchildval = thespinnernumchild.getValue().toString();
			bstaticsearchval = staticsearchcheck.isSelected();
			bpenalizedmodelval = penalizedButton.isSelected();
			ballowmergeval = allowmergecheck.isSelected();
			badd0 = nonormButton.isSelected();
			szevidenceval = evidenceField.getText().trim();
			sztaxonval = taxonField.getText().trim();
			bpontoval = pcheck.isSelected();
			bcontoval = ccheck.isSelected();
			bfontoval = fcheck.isSelected();
			bstaticcheckval = staticcheck.isSelected();
			brandomgoval = randomgoButton.isSelected();
			bmaxminval = maxminButton.isSelected();

			balltime = theRepeatList.allButton.isSelected();
			balltimemiRNA = miRNARepeatList.allButton.isSelected();
                        balltimeprote=ProteRepeatList.allButton.isSelected();
			System.out.println(szcommand);

			checkStatusTF = TfCheckBox.isSelected();
			checkStatusmiRNA = miRNACheckBox.isSelected();
			filtermiRNAExp = filtermiRNABox.isSelected();
			miRNATakeLog = miRNAlognormButton.isSelected();
			miRNAAddZero = miRNAnonormButton.isSelected();
			miRNAWeight = (Double) spinnerMIRNAWeight.getValue();
			dProbBindingFunctional = (Double) spinnerdProbBind.getValue();
			tfWeight = (Double) spinnerTFWeight.getValue();
			miRNAInteractionDataFile = miRNAInteractionField.getText();
			miRNAExpressionDataFile = miRNAExpressionField.getText();
			fastaFile = fastaDataField.getText();
			decodPath = decodPathField.getText();
			regScoreFile = regScoreField.getText();

			this.setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));

			ntodownload = 0;
			nexceptions = 0;
			final boolean bgetxref = xrefcheck.isSelected();
			final boolean bgetgoann = anncheck.isSelected();
			if (bgetgoann) {
				synchronized (lockpd) {
					ntodownload++;
					bdownloading[GOANN] = true;
					bexception[GOANN] = false;
					npercentdone[GOANN] = 0;
				}
				Runnable runpd = new Progressdownload(this, GOANN, szgoval);
				(new Thread(runpd)).start();
			}

			if (bgetxref) {
				synchronized (lockpd) {
					ntodownload++;
					bdownloading[XREF] = true;
					bexception[XREF] = false;
					npercentdone[XREF] = 0;
				}
				Runnable runpd = new Progressdownload(this, XREF, szxrefval);
				(new Thread(runpd)).start();
			}

			if (obocheck.isSelected()) {
				synchronized (lockpd) {
					ntodownload++;
					bdownloading[OBO] = true;
					bexception[OBO] = false;
					npercentdone[OBO] = 0;
				}

				Runnable runpd = new Progressdownload(this, OBO,
						szgocategoryval);
				(new Thread(runpd)).start();
			}

			if (bgetmirnaexp) {
				synchronized (lockpd) {
					ntodownload++;
					bdownloading[MIRNAEXP] = true;
					bexception[MIRNAEXP] = false;
					npercentdone[MIRNAEXP] = 0;
				}
				Runnable runpd = new Progressdownload(this, MIRNAEXP,
						miRNAInteractionDataFile);
				(new Thread(runpd)).start();
			}

			final JFrame fframe = this;

			Runnable clusterrun = new Runnable() {
				public void run() {
					clusterAButton.setEnabled(false);
					try {
						clusterscript(szstaticFileval,szxrefval, szorig1val,
								szgoval, szgocategoryval, szmaxmissingval,
								szexpressval, szfilterthresholdval,
								szsamplepvalval, szmingoval, szmingolevelval,
								szextraval, balltime, theRepeatList.data,
								miRNARepeatList.data,ProteRepeatList.data, btakelog, bgetxref,
								bgetgoann, bspotincluded, badd0,
								szcategoryIDval, szinitfileval, szevidenceval,
								sztaxonval, bpontoval, bcontoval, bfontoval,
								brandomgoval, bmaxminval);
					} catch (IllegalArgumentException iex) {
						final IllegalArgumentException fiex = iex;
						iex.printStackTrace(System.out);

						if (executeDialognf != null) {
							executeDialognf.setVisible(false);
							executeDialognf.dispose();
						}

						javax.swing.SwingUtilities.invokeLater(new Runnable() {
							public void run() {
								JOptionPane.showMessageDialog(fframe, fiex
										.getMessage(), "Error",
										JOptionPane.ERROR_MESSAGE);
							}
						});
					} catch (Exception ex) {
						final Exception fex = ex;

						if (executeDialognf != null) {
							executeDialognf.setVisible(false);
							executeDialognf.dispose();
						}

						javax.swing.SwingUtilities.invokeLater(new Runnable() {
							public void run() {
								JOptionPane.showMessageDialog(fframe, fex
										.toString(), "Exception thrown",
										JOptionPane.ERROR_MESSAGE);
								fex.printStackTrace(System.out);
							}
						});
					}
					clusterAButton.setEnabled(true);
					fframe.setCursor(Cursor
							.getPredefinedCursor(Cursor.DEFAULT_CURSOR));
				}
			};
			(new Thread(clusterrun)).start();
		} else if (esource == infoButton) {
			String szMessage = "iDREM: This is the interactive version of the Dynamic Regulatory Events Miner (DREM).\n\n"
					+ "The Original Dynamic Regulatory Events Miner (DREM) was developed by Jason Ernst "
					+ "in collaboration with Ziv Bar-Joseph and extended by William E. Devanny, Anthony Gitter, Marcel H. Schulz.\n\n"
                                        + "This iDREM (implemented by Jun Ding) extends DREM with an interfactive visualization "
                                        + "and also enables the integration of many additional information, such as epigenomics and proteomics. \n"
					+ "Any questions or bugs found should "
					+ "be emailed to {zivbj/jund}@cs.cmu.edu.";

			Util.renderDialog(this, szMessage, 50, 100, "Information");
		} else if ((numchildHButton == esource)
				|| (modelframeworkHButton == esource)
				|| (convergencePanelHButton == esource)
				|| (minstddevPanelHButton == esource)
				|| (nodepenaltyHButton == esource)
				|| (mergepathHButton == esource)
				|| (delaypathHButton == esource) || (epsilonHButton == esource)
				|| (extraHButton == esource)
				|| (staticsourceHButton == esource)
				|| (filterchoiceHButton == esource)
				|| (initfileHButton == esource)
				|| (initsearchHButton == esource)
				|| (mingolevelHButton == esource)
				|| (prunepathHButton == esource) || (seedHButton == esource)
				|| (staticFileHButton == esource)
				|| (allowmergeHButton == esource)
				|| (staticsearchHButton == esource)
				|| (threewayHButton == esource) || (staticHButton == esource)
				|| (viewstaticHButton == esource)
				|| (randomgoHButton == esource) || (evidenceHButton == esource)
				|| (ontoHButton == esource) || (taxonHButton == esource)
				|| (categoryIDHButton == esource) || (orig1HButton == esource)
				|| (orig2HButton == esource) || (advancedHButton == esource)
				|| (goLabelHButton == esource)
				|| (numberProfileHButton == esource)
				|| (changeHButton == esource) || (maxmissingHButton == esource)
				|| (filterthresholdHButton == esource)
				|| (expressHButton == esource) || (corrmodelHButton == esource)
				|| (alphamodelHButton == esource) || (lbHButton == esource)
				|| (percentileHButton == esource)
				|| (samplegeneHButton == esource)
				|| (samplemodelHButton == esource)
				|| (samplepvalHButton == esource) || (mingoHButton == esource)
				|| (mingolevelHButton == esource) || (methodHButton == esource)
				|| (logHButton == esource)
				|| (downloadlistgoHButton == esource)
				|| (xrefHButton == esource) || (presetsHButton == esource)
				|| (executeHButton == esource)
				|| (xrefsourceHButton == esource) || (viewHButton == esource)
				|| (spotHButton == esource) || (extraHButton == esource)
				|| (miRNAComboBoxHButton == esource)
				|| (staticmiRNAInterHButton == esource)
				|| (staticmiRNAExpHButton == esource)
				|| (miRNARepeatHButton == esource)
				|| (miRNANormHButton == esource)
				|| (miRNAScoringHButton == esource)
				|| (miRNAFilterHButton == esource)
				|| (miRNAWeightHButton == esource)
				|| (miRNATFWeightHButton == esource)
				|| (fastaDataHButton == esource)
				|| (decodPathHButton == esource)
				|| (regScoreHButton == esource)
                                || (ProteHButton== esource)// ProteHButton Reponse
                                || (ProteHCButton== esource) // ProteHCButton Response 
                                || (ProteHRButton == esource ) //ProteHRButton Response
                                || (ProteHPPIButton==esource)  //ProteHPPIButton response
                                || (proteweightHButton==esource) //proteweightHButton response
                                || (epigeneticweightHButton==esource) //epigeneticweightHButton response
                                || (MethyHButton==esource)
                                || (MethyGTFHButton==esource)
                                    ) {  
			makeHelpDialog((Component) esource);
		}
	}

	/**
	 * Download the data at the location of szURL into a file szoutfile ntype is
	 * used for updating the download progress percentages
	 */
	public boolean getFile(String szURL, String szoutfile, int ntype) {
		final JFrame fframe = this;
		try {
			URL theURL = new URL(szURL);
			URLConnection theurlc = theURL.openConnection();

			int ntotal = theurlc.getContentLength();

			InputStream stream = theurlc.getInputStream();
			BufferedInputStream in = new BufferedInputStream(stream);
			FileOutputStream file = new FileOutputStream(szoutfile);
			BufferedOutputStream out = new BufferedOutputStream(file);
			int ni;
			int nread = 0;
			npercentdone[ntype] = 0;
			byte[] b = new byte[1024];
			int nlastpercentdone = 0;
			while ((ni = in.read(b)) != -1) {
				nread += ni;
				double dpercent = ((double) nread) / ntotal;

				synchronized (lockpd) {
					npercentdone[ntype] = (int) (100 * dpercent);
					if (nlastpercentdone != npercentdone[ntype]) {
						nlastpercentdone = npercentdone[ntype];
						lockpd.notifyAll();
					}
				}
				out.write(b, 0, ni);
			}
			out.flush();
			stream.close();
			in.close();
			file.close();
			out.close();
			return true;
		} catch (final IOException fex) {
			synchronized (lockpd) {
				nexceptions++;
				bexception[ntype] = true;
				bdownloading[ntype] = false;
				lockpd.notifyAll();
				if (nexceptions == 1) {
					javax.swing.SwingUtilities.invokeLater(new Runnable() {
						public void run() {
							JOptionPane.showMessageDialog(fframe, fex
									.toString(), "Exception thrown",
									JOptionPane.ERROR_MESSAGE);
						}
					});
				}
			}
			return false;
		}
	}

	// ////////////////////////////////////////////////////////////////////
	/**
	 * A runnable class used for controlling the downloading of files
	 */
	public class Downloadfile implements Runnable {
		int ntype;
		String szfile;

		/**
		 * Constructor
		 */
		Downloadfile(int ntype, String szfile) {
			this.ntype = ntype;
			this.szfile = szfile;
		}

		/**
		 * Calls getFile with properly formatted URL to download files
		 */
		public void run() {
			String szurl;
			final String szfilef = szfile;

			if (ntype == GOANN) {
			    if (szfile.equals("goa_arabidopsis.gaf.gz"))
				{
				    szurl = EBIURL+"ARABIDOPSIS/"+szfile;
				}
			    else if (szfile.equals("goa_chicken.gaf.gz"))
				{
				    szurl = EBIURL+"CHICKEN/"+szfile;
				}
			    else if (szfile.equals("goa_cow.gaf.gz"))
				{
				    szurl = EBIURL+"COW/"+szfile;
				}
			    else if (szfile.equals("goa_dog.gaf.gz"))
				{
				    szurl = EBIURL+"DOG/"+szfile;
				}
			    else if (szfile.equals("goa_fly.gaf.gz"))
				{
				    szurl = EBIURL+"FLY/"+szfile;
				}
			    else if (szfile.equals("goa_human.gaf.gz"))
				{
				    szurl = EBIURL+"HUMAN/"+szfile;
				}
			    else if (szfile.equals("goa_mouse.gaf.gz"))
				{
				    szurl = EBIURL+"MOUSE/"+szfile;
				}
			    else if (szfile.equals("goa_pdb.gaf.gz"))
				{
				    szurl = EBIURL+"PDB/"+szfile;
				}
			    else if (szfile.equals("goa_pig.gaf.gz"))
				{
				    szurl = EBIURL+"PIG/"+szfile;
				}
			    else if (szfile.equals("goa_rat.gaf.gz"))
				{
				    szurl = EBIURL+"RAT/"+szfile;
				}
			    //else if (szfile.equals("goa_uniprot_all.gaf.gz"))
			    //{
			    //   szurl = EBIURL+"UNIPROT/"+szfile;
			    //}
			    else if (szfile.equals("goa_worm.gaf.gz"))
				{
				    szurl = EBIURL+"WORM/"+szfile;
				}
			    else if (szfile.equals("goa_yeast.gaf.gz"))
				{
				    szurl = EBIURL+"YEAST/"+szfile;
				}
			    else if (szfile.equals("goa_zebrafish.gaf.gz"))
				{
				    szurl = EBIURL+"ZEBRAFISH/"+szfile;
				}
			    else
				{
				    //szurl ="ftp://ftp.geneontology.org/go/gene-associations/"+szfile;
				    //changed in 1.3.6 to access http site instead of ftp
				    szurl ="http://www.geneontology.org/gene-associations/"+szfile;
				}

			} else if (ntype == XREF) {
				szurl = EBIURL;
				if (szfile.equals("chicken.xrefs.gz")) {
					szurl += "CHICKEN/";
				}

				if (szfile.equals("human.xrefs.gz")) {
					szurl += "HUMAN/";
				} else if (szfile.equals("mouse.xrefs.gz")) {
					szurl += "MOUSE/";
				} else if (szfile.equals("rat.xrefs.gz")) {
					szurl += "RAT/";
				} else if (szfile.equals("arabidopsis.xrefs.gz")) {
					szurl += "ARABIDOPSIS/";
				} else if (szfile.equals("zebrafish.xrefs.gz")) {
					szurl += "ZEBRAFISH/";
				}
				szurl += szfile;
			} else if (ntype == OBO) {
				szurl = "http://www.geneontology.org/ontology/gene_ontology.obo";
			} else { // ntype is a MIRNAEXP file type
				szurl = "MIRNA DATA WEBSITE" + szfile;
			}//TODO: replace with hosting website

			final String szurlf = szurl;

			getFile(szurlf, szfilef, ntype);

			synchronized (lockpd) {
				ntodownload--;
				bdownloading[ntype] = false;
				lockpd.notifyAll();
			}
		}
	}

	/**
	 * Runnable class used to display the download progress and also starts the
	 * download of a file
	 */
	public class Progressdownload implements Runnable {
		String szfile;
		int ntype;
		JFrame theframe;

		/**
		 * Constructor
		 */
		Progressdownload(JFrame theframe, int ntype, String szfile) {
			this.theframe = theframe;
			this.ntype = ntype;
			this.szfile = szfile;
		}

		/**
		 * The run method starts the download of the file and updates the
		 * download progress
		 */
		public void run() {
			// launch download thread
			Runnable rundf = new Downloadfile(ntype, szfile);
			(new Thread(rundf)).start();
			final int noffsetxf = 250;
			final int noffsetyf;
			final int ntypef = ntype;

			if (ntypef == GOANN) {
				noffsetyf = 150;
			} else if (ntypef == XREF) {
				noffsetyf = 250;
			} else if (ntypef == OBO) {
				noffsetyf = 350;
			} else { // ntypef == MIRNAEXP
				noffsetyf = 450;
			}
			final JProgressBar thebar = new JProgressBar(0, 100);

			final JDialog progressDialog;
			if (ntypef == GOANN) {
				progressDialog = new JDialog(theframe,
						"Gene Annotation File Download Progress");
			} else if (ntypef == XREF) {
				progressDialog = new JDialog(theframe,
						"Cross Reference File Download Progress");
			} else if (ntypef == OBO) {
				progressDialog = new JDialog(theframe,
						"Ontology File Download Progress");
			}  else { // ntypef == MIRNAEXP
				progressDialog = new JDialog(theframe,
						"miRNA File Download Progress");
			}
			progressDialog.setCursor(Cursor
					.getPredefinedCursor(Cursor.WAIT_CURSOR));

			javax.swing.SwingUtilities.invokeLater(new Runnable() {
				public void run() {
					synchronized (lockpd) {
						npercentdone[ntype] = 0;
						thebar.setValue(0);
						if (!bexception[ntype]) {
							Container theprogressDialogPane = progressDialog
									.getContentPane();
							progressDialog.setSize(400, 75);
							thebar.setSize(400, 75);
							thebar.setStringPainted(true);
							progressDialog.setBackground(Color.white);
							theprogressDialogPane.setBackground(Color.white);
							theprogressDialogPane.add(thebar);
							progressDialog.setLocation(theframe.getX()
									+ noffsetxf, theframe.getY() + noffsetyf);
							progressDialog
									.setDefaultCloseOperation(DISPOSE_ON_CLOSE);
							progressDialog.setVisible(true);
						}
					}
				}
			});

			synchronized (lockpd) {
				bdownloading[ntype] = true;
				while (bdownloading[ntype]) {
					try {
						lockpd.wait();
					} catch (InterruptedException ex) {
						ex.printStackTrace(System.out);
					}
					thebar.setValue(npercentdone[ntype]);
				}
			}

			if ((!bexception[ntype]) && (ntype == GOANN)) {
				anncheck.setEnabled(true);
				anncheck.setSelected(false);
			} else if ((!bexception[ntype]) && (ntype == XREF)) {
				xrefcheck.setEnabled(true);
				xrefcheck.setSelected(false);
			} else if ((!bexception[ntype]) && (ntype == OBO)) {
				obocheck.setEnabled(true);
				obocheck.setSelected(false);
			} else if ((!bexception[ntype])) {// ntype is a MIRNAEXP
//				mirnaexpcheck.setEnabled(true);
//				mirnaexpcheck.setSelected(false);
			}
			progressDialog.setVisible(false);
			progressDialog.dispose();
			progressDialog.setCursor(Cursor
					.getPredefinedCursor(Cursor.DEFAULT_CURSOR));
		}
	}

	// ////////////////////////////////////////////////////
	/**
	 * Renders a help dialog based on the specific help button pressed
	 */
	public void makeHelpDialog(Component esource) {

		String szMessage = "";

		if (esource == modelframeworkHButton) {
			szMessage = "Two frameworks, 'Penalized Likelihood' and 'Train-Test' "
					+ "for model selection are available.   "
					+ "Under the 'Penalized Likelihood' option all the genes are used to both train the parameters of "
					+ "the model during search and select the model.  "
					+ "A regularization parameter, 'Penalized Likelihood Node Penalty', "
					+ "is the penalty subtracted for each state to prevent overfitting. "
					+ "Under the train-test option, a subset of genes are used to train "
					+ "the parameters of the model under consideration, "
					+ "while the remainder are used to score the model based on likelihood.  "
					+ "A second phase is then executed under this option where the data is resplit "
					+ "and only changes which reduce the number of states are considered.  "
					+ "The nine parameters at the bottom of this panel "
					+ "become active when this model selection criteria is used.";
			Util.renderDialog(theOptions, szMessage, 50, 100);
                } else if (esource==MethyHButton){
                        szMessage="epigenomic data:\n"
                                + "This epigenomic data includes the data like DNA methylation, histone methylation, etc. \n"
                                + "Here the epigenomic score is used to denote the repression of the region.\n"
                                + "Therefore, different types of epigenomic data need to be pre-processed differently. \n\n"
                                + "For example, if the epigenomic data is DNA methylation, the normalized methylation score [0-1] can be used directly as the input. "
                                + "If the epigenomic data is histone modification, e.g H3K4me2, which is asssociated with activiation,"
                                + "then the input should be 1-normalized_histone_modification_score. \n"
                                + "In short, the epigenomic data here should represent 'difficulty' score of TF-binding. The larger score, the smaller "
                                + "probablity of TF binding\n\n"
                                + "If having replicates for each time point, the mean epigenomic score will be used.\n"
                                + "The epigenomic data input should be in BED6 format.\n"
                                + "This file has the following BED6 formatting requirements: \n"
                                + "1st column: chrom \n"
                                + "2nd column: ChromStart \n"
                                + "3rd column: ChromEnd \n"
                                + "4th column: Name with time point information. It should be in the format of TimePoint_Gene \n"
                                + "5th column: epigenetic score (in [0,1]) \n"
                                + "6th column: strand \n"
                                + "\n\n"
                                +"SAMPLE File:\n"
                                +"chr7	28372162	28373662	p0.5_Plekhg2	0.21	-\n" +
                                "chr12	76532560	76534060	p0.5_Plekhg3	0.25	+\n" +
                                "chr10	3739377	3740877	p0.5_Plekhg1	0.56	+\n" +
                                "chr6	125380004	125381504	p0.5_Plekhg6	0.41	-\n" 
                               
                                +"\n\n\n";
                        Util.renderDialog(theOptions,szMessage,50,100);
                } else if (esource==MethyGTFHButton){
                        szMessage="This is the GTF file associated with given organisms.\n"
                                +"The gene annotation will be obtained from the given GTF file.\n"
                                +"For GTF format, plese refers to : http://www.ensembl.org/info/website/upload/gff.html";
                                   
                        Util.renderDialog(theOptions, szMessage,50,100);
                } else if (esource==ProteHButton){
                        // ProteHButton Action
                        szMessage = "This entry specifies a file that contains the time-series proteomics data. A data file "
                                +"includes gene symbols, data values. This file has the following formatting requirements: \n"
                                +"(1) The first row specifies the time points.\n"
                                +"(2) For every row after the first, 1st Columns tells the gene name, "
                                + "the following columns tell the corresponding protein level of the gen at each time point. \n "
                                +"\n"
                                +"SAMPLE File:\n"
                                +"Name\te16.5\tp0.5\tp1\tp1.5\tp2.5\n"
                                +"Pabpn\t1.26\t0.42\t0.0\t0.0\t0.47";
                        Util.renderDialog(theOptions, szMessage,50,100);
                } else if (esource==proteweightHButton){
                    //proteomics data weight H button
                    szMessage="This entry specifies the relative weight of the proteomics data to epigenomics data on inferring dynamic (time-depedent) TF-gene interaction map. \n "
                            +"The proteomics and epigenetics data contribute to the model indirectly (via the impact on dynamic TF-gene map). \n"
                            +"This relative weights species the realtive importance of each data type\n"
                            +"By default, the protoemics data and epigenomics data are treated with equal importance (weight =1).\n "
                            +"Users can adjust the weights if they have the prior knowledge that one data is more influencial than the other.\n"
                            +"For example, set relative weight <1 to emphasize more on epigenetics data and set relative weight >1 to emphasize more \n"
                            + "on the proteomics data.";
                    Util.renderDialog(theOptions, szMessage,50,100);
                } else if (esource==ProteHCButton){
                        //ProteHCButton Action
                        szMessage = "Choose how to use the proteomics data in the following 3 ways: \n"
                                + "(1) Only used proteomics data for TFs.\n"
                                + "(2) Used proteomics data for all proteins, including TFs and interacting proteins.\n"
                                +"(3) Do not use proteomics data.\n";
                        Util.renderDialog(theOptions,szMessage,50,100);
                } else if (esource==ProteHPPIButton){
                        szMessage="This entry specifies the Protein-protein interaction file. \n"
                                + "Such data can be downloaded from PPI databases such as STRING or BioGRID. "
                                + "This file has the following formatting requirements: \n\n"
                                + "First ,Second Columns present the interacting protein pairs (Using gene names). "
                                + "The first column tells the interation strength. If such information is not avaiable, use 1 instead."
                                +"\n"
                                +"SAMPLE File:\n"
                                +"Gnai3	Adora3	0.813\n" +
                                "Gnai3	Grm6	0.902\n" +
                                "Gnai3	Bdkrb2	0.904\n" +
                                "Gnai3	Akt1	0.717\n" +
                                "Gnai3	Adcy4	0.966\n" +
                                "Gnai3	Rgs19	0.987\n" +
                                "Gnai3	Hck	0.889\n\n\n";
                        
                        Util.renderDialog(theOptions,szMessage,50,100);
                } else if (esource == ProteHRButton) {
                        //ProteHRButton Action
                        szMessage =     "The proteomics data normalization:\n"
                                        +"All time series will be transformed so that the "
					+ "time series starts at 0.  "
					+ "This can be done in one of three ways based on the option selected to the left.  "
					+ "Given a time series vector of values for a gene (v_0,v_1,...,v_n) the options are:\n"
					+ "1.  'Log normalize data'  \u2212  the vector "
					+ "will be transformed to (0,log\u2082(v_1)\u2212log\u2082(v_0),...,log\u2082(v_n)\u2212log\u2082(v_0)).  "
					+ "Note that any values which are 0 or negative will be treated as missing.\n"
					+ "2.  'Normalize data'  \u2212  the vector "
					+ "will be transformed to (0,v_1\u2212v_0,...,v_n\u2212v_0)\n"
					+ "3.  'No normalization/add 0'  \u2212  a 0 will be inserted transforming the vector to "
					+ "(0,v_0,v_1,...,v_n)\n\n"
					+ "*If the data is not already in log space as often is the case if it is "
					+ "from an Oligonucleotide array, then the "
					+ "'Log normalize data' should be selected.\n"
					+ "*If the data is already in log space as is often the case if the data is "
					+ "from a two channel cDNA array and "
					+ "a time point 0 experiment was "
					+ "conducted, then the 'Normalize data' option should be selected.\n"
					+ "*If the data is already in log space and no time point 0 experiment was "
					+ "conducted, then the 'No normalization/add 0' option should be selected.";
                        Util.renderDialog(theOptions,szMessage,50,100);
		} else if (esource == convergencePanelHButton) {
			szMessage = "This parameter controls the percentage likelihood gain required "
					+ "to continue searching for better parameters for the model. "
					+ "Increasing this parameter can lead to a faster running time, "
					+ "decreasing it may lead to better values of the parameters.";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == minstddevPanelHButton) {
			szMessage = "This parameter controls the minimum standard deviation on the Gaussian distributions. "
					+ "Increasing this parameter is recommended if applying DREM to RNA-seq data to avoid potential overfitting of "
					+ "low variance in expression due to small discrete counts.";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == nodepenaltyHButton) {
			szMessage = "This parameter is only active if the 'Penalized Likelihood' "
					+ "option is selected under the model selection framework, in which case "
					+ "it is the penalty for each node (state) in the final model.  Increasing "
					+ "the parameter would bias DREM to select a model with fewer nodes, while "
					+ "decreasing it will cause more nodes.";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == ontoHButton) {
			szMessage = "These three checkboxes allow one to filter annotations that are not "
					+ "of the types checked.  These three checkboxes only apply if the annotations are in the official "
					+ "15 column GO format, in which case the annotation type is determined by "
					+ "the entry in the 'Aspect' field (Column 9).  "
					+ "An entry of 'P' in the 'Aspect' field means the annotation is of type "
					+ "'Biological Process', an entry of 'F' "
					+ "means the annotation is of type 'Molecular Function', and an entry of 'C' means the annotation "
					+ "is of type 'Cellular Component'.";

			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == taxonHButton) {
			szMessage = "Some annotation files contain annotations for multiple "
					+ "organism, and it might be desirable to use annotations only for certain organisms.  "
					+ "To use only annotations for "
					+ "certain organisms enter the taxon IDs for the desired organisms delimited "
					+ "by either commas (','), semicolons (';'), or pipes ('|').  "
					+ "If this field is left empty, then any organism is assumed to be acceptable.  More information "
					+ "about taxonomy codes and a search function to find the taxon code for an organism can be found at "
					+ "http://www.ncbi.nlm.nih.gov/Taxonomy/.  Note that this parameter only applies when "
					+ "the annotations are in the official 15 column format.  "
					+ "The taxonomy ID in the annotation file is in column 13 of the file, and the taxon "
					+ "IDs entered in this parameter field must match the entry in column 13 or match after "
					+ "prepending the string 'taxon:' to the ID.  For example to use only annotations "
					+ "for a 'Homo sapien' the string '9606' can be used.";

			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == evidenceHButton) {
			szMessage = "This field takes a list of unacceptable evidence codes for "
					+ "gene annotations delimited by either a comma (','), semicolon (';'), "
					+ "or pipe ('|').  If this field is left empty, then "
					+ "all evidence codes are assumed to be acceptable.  Evidence code symbols are "
					+ "IEA, IC, IDA, IEP, IGI, IMP, IPI, ISS, RCA, NAS, ND, TAS, and NR.  "
					+ "Information about GO evidence codes can be found at "
					+ "http://www.geneontology.org/GO.evidence.codes.shtml.  Note that this field only applies "
					+ "if the gene annotations are in the official 15 column GO annotation format.  "
					+ "The evidence code is the entry in column 7.  "
					+ "For example to exclude the annotations that were inferred from electronic annotation "
					+ "or a non-traceable author statement the field should contain IEA;NAS.";

			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == viewHButton) {
			szMessage = "Pressing the 'View Expression Data' button opens a table with "
					+ "the contents of the file listed under 'Expression Data File'.";
			Util.renderDialog(this, szMessage, 50, 100);
		} else if (esource == allowmergeHButton) {
			szMessage = "If checked then paths sharing a common prior split are also to be modeled to merge.  "
					+ "If unchecked then once paths split they will not be modeled to merge again.";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == staticFileHButton) {
			szMessage = "File encodes the static transcription factor-gene interaction predictions as "
					+ "input to DREM.  The file is a tab delimited text file, that can be in one of "
					+ "two formats.  Either in a grid format or in a three column format.\n\n"
					+ "The grid format file is as follows:\n"
					+ "The first column of the file contains the gene identifier symbols.  "
					+ "The first row contains the transcription factors identifiers.  "
					+ "The first entry of the first row is a label for the gene symbol column.  "
					+ "Each remaining entry corresponds to the relationship between a transcription factor and a "
					+ "gene.  Under a binary encoding an entry is 1 if the transcription factor is predicted "
					+ "to regulate the gene and 0 otherwise.  If a three way encoding is used an entry is 1 if "
					+ "the transcription factor is predicted to activate the gene, -1 if it is predicted to repress "
					+ "the gene and 0 otherwise. Below is portion of a sample TF-gene interaction input file\n\n"
					+ "ID	ADR1	ARG80	ARG81	ARO80	BAS1	CAD1	CBF1\n"
					+ "YAL053W	0	0	0	0	0	0	1\n"
					+ "YAL054C	0	0	0	0	0	0	1\n"
					+ "YAL055W	0	0	0	0	1	0	0\n\n"
					+ "In the three column format the first column contains "
					+ "the transcription factors, the second column "
					+ "the regulated gene, and the third column input value.  The first row is a header row "
					+ "where the header of the first column must be 'TF' column, "
					+ "and the second column must have the header 'Gene'.  "
					+ "If a TF-gene pair is not present "
					+ "the input value is assumed to be 0.  When there are a lot of TFs and genes with a sparse number of "
					+ "non-zero entries then the three column format can lead to significant savings in space.\n"
					+ "TF	Gene	Input\n"
					+ "BAS1	YAL055W	1\n"
					+ "CBF1	YAL053W	1\n" + "CBF1	YAL054C	1\n";
			Util.renderDialog(this, szMessage, 50, 100);
		} else if (esource == mergepathHButton) {
			szMessage = "These parameters control the merging of paths if 'Allow Path Merges' is selected on "
					+ "the search options.  Increasing the percent or decreasing the threshold will cause more "
					+ "paths to be merged.  The percentage parameter must be >=0 and the threshold parameter must be less <=0.";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == delaypathHButton) {
			szMessage = "These parameters control the delaying of splits.  "
					+ "Increasing the percent or decreasing the threshold will cause more "
					+ "splits to be delayed.  The percentage parameter must be >=0 and the threshold parameter must be less <=0.";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == epsilonHButton) {
			szMessage = "This parameter specifies the score improvement required for DREM to continue considering "
					+ "to add paths.  Increasing the percent or the threshold will cause DREM to terminate it search sooner. "
					+ "The percentage parameter must be >=0 and the threshold parameter must be less >=0";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == staticsourceHButton) {
			szMessage = "This menu lists the files in the TFInput directory.  Selecting a file "
					+ "from the menu enters that file name in the 'TF-gene Interactions File' field.  "
					+ "Selecting 'User Provided' leaves the 'TF-gene Interactions File' field empty so "
					+ "that a user can specify the desired interaction file.  The included files are for yeast based "
					+ "on Harbison et al, Nature 2004 and MacIsaac et al, BMC Bioinformatics 2006; for E. coli "
					+ "curated direct evidence inputs based on EcoCyc 11.5 and those interactions extened with "
					+ "additional predictions as described in Ernst et al, PLoS CompBio 2008."
					+ "Consult Appendix B of the user manual for their description.  ";
			Util.renderDialog(this, szMessage, 50, 100);
		} else if (esource == filterchoiceHButton) {
			szMessage = "The 'Change should be based on parameter' defines how change is defined in the context "
					+ "of gene filter.  "
					+ "If 'Maximum\u2212Minimum' option is selected a gene will be filtered if the maximum "
					+ "absolute difference between "
					+ "the values of any two time points, not necessarily consecutive, "
					+ "after transformation is less than the value of "
					+ "the 'Minimum Absolute Expression Change' parameter.  "
					+ "If 'Difference from 0' is selected a gene "
					+ "will be filtered if the absolute expression change from time point 0 at all time points is less than the "
					+ "value of the 'Minimum Absolute Expression Change' parameter.\n\n"
					+ "Formally suppose (0,v_1,v_2,...,v_n) is the expression level of a gene "
					+ "after transformation and "
					+ "let C be the value of the 'Minimum Absolute Expression Change'.  "
					+ "If the 'Maximum-Minimum' option is selected a gene will be filtered if "
					+ "max(0,v_1,v_2,...v_n)\u2212min(0,v_1,v_2,...,v_n)<C.  If the "
					+ "'Minimum Absolute Expression Change' option is selected "
					+ "the gene will be filtered "
					+ "if max(0,|v_1|,|v_2|,...,|v_n|)<C.\n\n"
					+ "Only the 'Maximum\u2212Minimum' option guarantees that the "
					+ "same set of genes would be filtered for any permutation of the time points.  "
					+ "For the 'Difference from 0' this is in general not true, in this case the permutation "
					+ "test is based on the set of genes passing filter under the original order of time points.";

			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == initfileHButton) {
			szMessage = "This file is optional and encodes a saved model produced by DREM.  "
					+ "A model produced by DREM can be saved by pressing the button 'Save Model' "
					+ "on the main DREM interface window.  Depending on the setting of 'Saved Model' under "
					+ "'Search Options' this model is either opened as is, a search is started from it, or is "
					+ "not used.";
			Util.renderDialog(this, szMessage, 50, 100);
		} else if (esource == initsearchHButton) {
			szMessage = "This option is only relevant if a file is specified under 'Saved Model File'.  "
					+ "If it is set to 'Use As Is' the model in the 'Saved Model File' is opened exactly as is,  "
					+ "if the parameter is set to 'Start Search From' DREM will start its search from the model saved "
					+ "in 'Saved Model File', if the parameter is set to 'Do Not Use' then DREM will start a new search "
					+ "for a model regardless what is specifiec in 'Saved Model File'.";

			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == prunepathHButton) {
			szMessage = "These parameters control the deleting of paths during the final phase of the DREM algorithm.  "
					+ "Increasing the percent or decreasing the threshold parameters will cause more "
					+ "paths to be removed.  The percentage parameter must be >=0 and the threshold parameter must be less <=0.";

			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == seedHButton) {
			szMessage = "This parameter is the random seed used by DREM when "
					+ "randomly partitioning the data set into a training set and test "
					+ "set.  Changing the value of this parameter can result in different, but "
					+ "will usually share many common features.";

			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == staticsearchHButton) {
			szMessage = "If this box is checked then the transcription "
					+ "factor-gene interaction data is used jointly with the time series data to infer the model and then assign genes "
					+ "to paths of the model. If this box is unchecked then the time series data alone is used to infer a model, and "
					+ "the transcription factor-gene interaction predictions are only used as a post-processing step which scores "
					+ "TFs with splits and paths based on the gene assignments. Using the TF-gene interaction data to infer the "
					+ "model generally gives a more biologically coherent model. Using the TF-gene information only as a "
					+ "post-processing step, the TF-gene scores can be interpreted directly as p-values, which is not the case when "
					+ "the box is checked. Also learning a model is faster when not using the TF-gene interaction data.";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == numchildHButton) {
			szMessage = "Determines the maximum number of paths out of any split.  ";

			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == categoryIDHButton) {
			szMessage = "This file, which is optional, specifies a mapping between category IDs and names.  "
					+ "The first column contains category IDs while the second column contains category names "
					+ "corresponding to the category ID in the first column.  Note that the category names for "
					+ "official Gene Ontology (GO) categories are included in the 'gene_ontology.obo' file "
					+ "and thus do not need to be included here. "
					+ "This file is rather intended to define names of additional gene sets "
					+ "that are not part of GO, but will be included in a GO analysis.  "
					+ "If no mapping is made between a category ID and category name, then the category ID "
					+ "is used in place of the category name.\n\n"
					+ "SAMPLE FILE:\n"
					+ "ID_A	Category A\n"
					+ "ID_B	Category B\n" + "ID_C	Category C\n"
                                        +"\n\n\n";

			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == spotHButton) {
			szMessage = "Each entry in the data file is associated with a unique identifier called a spot ID.  "
					+ "This identifier is different than the genes symbols which need not be unique.  "
					+ "Spot IDs can either be included in the data file as the first column or not included and then "
					+ "automatically generated.\n"
					+ "*If the 'Spot IDs included in the data file' box is checked then the first column of "
					+ "the data file contains the spot IDs "
					+ "and the second column contains gene symbols.\n"
					+ "*If the 'Spot IDs included in the data file' box is unchecked then  the first column of the data "
					+ "file contains gene symbols and the spot IDs are "
					+ "automatically generated for each entry in "
					+ "the data file.  "
					+ "Spot IDs are automatically generated as sequential integers starting at 0 "
					+ "appended to the string \"ID_\".\n";

			Util.renderDialog(this, szMessage, 50, 100);
		} else if (esource == logHButton) {
			szMessage = "All time series will be transformed so that the "
					+ "time series starts at 0.  "
					+ "This can be done in one of three ways based on the option selected to the left.  "
					+ "Given a time series vector of values for a gene (v_0,v_1,...,v_n) the options are:\n"
					+ "1.  'Log normalize data'  \u2212  the vector "
					+ "will be transformed to (0,log\u2082(v_1)\u2212log\u2082(v_0),...,log\u2082(v_n)\u2212log\u2082(v_0)).  "
					+ "Note that any values which are 0 or negative will be treated as missing.\n"
					+ "2.  'Normalize data'  \u2212  the vector "
					+ "will be transformed to (0,v_1\u2212v_0,...,v_n\u2212v_0)\n"
					+ "3.  'No normalization/add 0'  \u2212  a 0 will be inserted transforming the vector to "
					+ "(0,v_0,v_1,...,v_n)\n\n"
					+ "*If the data is not already in log space as often is the case if it is "
					+ "from an Oligonucleotide array, then the "
					+ "'Log normalize data' should be selected.\n"
					+ "*If the data is already in log space as is often the case if the data is "
					+ "from a two channel cDNA array and "
					+ "a time point 0 experiment was "
					+ "conducted, then the 'Normalize data' option should be selected.\n"
					+ "*If the data is already in log space and no time point 0 experiment was "
					+ "conducted, then the 'No normalization/add 0' option should be selected.";

			Util.renderDialog(this, szMessage, 50, 100);
		} else if (esource == orig1HButton) {
			szMessage = "This entry specifies a file that contains gene expression data.  "
					+ "A data file includes gene symbols, data values, and optionally spot IDs. "
					+ "Spot IDs uniquely identify an entry.  If spot IDs are included in the data file the field "
					+ "'Spot IDs included in the data file' must be checked, otherwise the field must be unchecked and "
					+ "IDs for each entry in the data file will be generated.  "
					+ "While spot IDs must be unique, the same gene symbol can appear multiple times in the data file.\n\n"
					+ "The file has the following "
					+ "formatting restrictions:\n"
					+ "* The first line contains the column headers delimited by tabs.\n"
					+ "* The remaining lines contain the spot IDs (optionally), "
					+ "gene symbols, and then data delimited by tabs.\n"
					+ "* If the 'Spot IDs included in the data file' box is checked, then the first column "
					+ "contains the spot IDs and each spot ID must be unique.\n"
					+ "* The next column, or the first column if the 'Spot IDs included in the data file' "
					+ "box is unchecked, contains the gene symbols.  If there is no gene symbol associated with a given spot,"
					+ " then the field can either contain a \"0\" or no entry.\n"
					+ "* In either the spot or gene field there can be multiple symbols listed delimited by "
					+ "either a pipe ('|'), comma (','), or a semi-colon (';').  "
					+ "For the purposes of gene annotations just one symbol needs to match an annotation, while "
					+ "for the purpose of determining if two spots or genes are the same the entire entry must match.\n"
					+ "* If multiple spots correspond to the same gene, "
					+ "then the data for that gene will be combined using the median values after "
					+ "normalization.\n"
					+ "* The remaining columns contain the expression values in sequential order of time for the gene.\n"
					+ "* If a value is missing between two time points then the field should be left empty giving "
					+ "two tabs between the non-missing values.\n\n"
					+ "SAMPLE FILE with spot IDs included:\n"
					+ "Spot	Gene	0h	1h	3h	6h	12h\n"
					+ "ID_1	YAL053W	-0.027	0.158	0.169	0.193	-0.165\n"
					+ "ID_2	YAL054C	0.183	-0.068	-0.134	-0.252	0.177\n"
					+ "ID_3	YAL055W	-0.923	-0.51	-0.718	-0.512	-0.668\n\n"
					+ "SAMPLE FILE without spot IDs included:\n"
					+ "Gene	0h	1h	3h	6h	12h\n"
					+ "YAL053W	-0.027	0.158	0.169	0.193	-0.165\n"
					+ "YAL054C	0.183	-0.068	-0.134	-0.252	0.177\n"
					+ "YAL055W	-0.923	-0.51	-0.718	-0.512	-0.668";

			Util.renderDialog(this, szMessage, 50, 100);
		} else if (esource == staticHButton) {
			szMessage = "If this box is checked then genes are filtered if they are not included in the "
					+ "transcription factor-gene interaction file.  "
					+ "If this box is unchecked then genes not included in the file "
					+ "are not filtered and are assumed "
					+ "to have a '0' for every entry.";

			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == viewstaticHButton) {
			szMessage = "Pressing the 'View TF-gene Data' button displays the content of the file "
					+ "listed under 'TF-gene Interactions File'.";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == extraHButton) {
			szMessage = "This file is optional.  If included any genes listed in the file "
					+ "will be considered part of the initial base set of genes during a GO analysis in  "
					+ "addition to any genes included in the data file. "
					+ " Using this file thus allows one to filter "
					+ "genes from the data set by a criteria not implemented in DREM by excluding them from the "
					+ "data file, but still include the "
					+ "filtered genes as part of the "
					+ "initial base set of genes during the GO analysis by including them in this file.  "
					+ "If genes appear in both this file and the data file the gene will only "
					+ "be added to the base set once.  "
					+ "The format of this file is the same as a data file, except including the expression values is "
					+ "optional and if included they will be ignored.  "
					+ "As with a data file the first column will contain spot IDs "
					+ "if the field 'Spot IDs included in the data file' is checked and the second column will contain "
					+ "gene symbols, otherwise the first column will contain gene symbols.";

			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == advancedHButton) {
			szMessage = "  Pressing the 'Options' button opens a dialog box with "
					+ "options pertaining to filtering genes, search, model selection, "
					+ "gene annotations, and the Gene Ontology (GO) enrichment analysis.";

			Util.renderDialog(this, szMessage, 50, 100);
		} else if (esource == randomgoHButton) {
			szMessage = "This parameter controls the correction method for actual size based GO enrichment.  "
					+ "Expected size based p-values are always "
					+ "corrected using a Bonferroni correction.  "
					+ "The parameter value can either be 'Bonferroni' or 'Randomization'.  "
					+ "If 'Bonferroni' is selected then a Bonferroni correction is applied where the uncorrected "
					+ "p-value is divided by the number of categories meeting the minimum "
					+ "'Minimum GO level' and 'Minimum number of genes' constraints.  "
					+ "If 'Randomization' is selected the corrected p-value is computed "
					+ "based on a randomization test where random samples "
					+ "of the same size of the set being analyzed is drawn.  "
					+ "The number of samples is specified by the parameter "
					+ "'Number of samples for multiple hypothesis correction'.  "
					+ "The corrected p-value for a p-value, r, is the proportion of random "
					+ "samples for which there is enrichment for any GO category with a p-value less than r.  "
					+ "A Bonferroni correction is faster, but a randomization test leads to lower p-values.";

			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == orig2HButton) {
			szMessage = "     Clicking on the 'Repeat Data' button brings up a dialog box "
					+ "to specify repeat data files of the experiment.  Repeat data files "
					+ "are optional.  The format of repeat data is the same as the original data.  "
					+ "If included the repeat data values will be averaged using the median with the "
					+ "original experiment values.  "
					+ "Repeat data can either represent repeat measurements taken concurrently with the "
					+ "original experiment, or distinct full repeat experiments taken at different time periods.  "
					+ "If the data is of the latter type genes will be filtered that do not display a consistent "
					+ "expression profile between repeats based on the 'Minimum Correlation between Repeats' "
					+ "parameter under the 'Filtering' panel on the options menu.  "
					+ "If the button is yellow then there is currently repeat data loaded, otherwise the button is gray.";

			Util.renderDialog(this, szMessage, 50, 100);
		} else if (esource == goLabelHButton) {
			szMessage = "This file contains the Gene Ontology (GO) annotations of genes.  "
					+ "The file can be in one of two formats:\n\n"
					+ "1.  The file can be in the official 15 column GO Annotation format "
					+ "described at http://www.geneontology.org/GO.annotation.shtml#file.  In this case any entry in the "
					+ "DB_Object_ID (Column 2), DB_Object_Symbol (Column 3), DB_Object_Name (Column 10), or "
					+ "DB_Object_Synonym (Column 11) fields matching a spot ID or gene symbol in the data set "
					+ "will be annotated as belonging to GO ID (Column 5).  "
					+ " If the entry in the 'DB_Object_Symbol' "
					+ "contains an underscore ('_'), then the portion of the entry before "
					+ "the underscore will also be annotated as "
					+ "belonging to the GO category since under some naming conventions "
					+ "the portion after the underscore is a symbol "
					+ "for the database that is not specific to the gene.  "
					+ "The 'DB_Object_Synonym' column may have multiple symbols delimited by either "
					+ "a semicolon (';'), comma (','), or a pipe ('|') symbol and all will be "
					+ "annotated as belonging to the GO category in Column 5.  "
					+ "Note that the exact content of the 'DB_Object_ID', 'DB_Object_Symbol', "
					+ "'DB_Object_Name', and 'DB_Object_Synonym' varies between annotation source, "
					+ "consult the README files available at http://www.geneontology.org/GO.current.annotations.shtml "
					+ "to find out more information about the content of these fields for a specific annotation source.\n\n"
					+ "2.  Alternatively the file can have two columns where the first column contains gene symbols "
					+ "or spot IDs and the second column contains annotations of the genes in the first column. "
					+ "The two columns are delimited by a tab.  "
					+ "Gene symbols and GO annotations can be delimited by "
					+ "either a semicolon (;), comma (','), or a pipe (|).\n\nNote:\n"
					+ "*If a gene is listed as belonging to a certain GO category that is a sub-category"
					+ " of other categories in the GO hierarchy, it is not necessary to also explicitly list "
					+ "its super-categories.\n"
					+ "*If the same gene appears on multiple lines the union of "
					+ "annotation terms is taken.\n"
					+ "*The file can either be in plain text or "
					+ "a gzipped version of a plain text file in the required format.\n\n"
					+ "Sample file of two column format:\n"
					+ "ZFX	GO:0003677;GO:0003713;GO:0008270;GO:0030528;GO:0046872;GO:0006355;GO:0005634\n"
					+ "ZNF133	GO:0003700;GO:0008270;GO:0006355;GO:0005634\n"
					+ "USP2	GO:0004197;GO:0004221;GO:0016787;GO:0006511";
			Util.renderDialog(this, szMessage, 50, 100);
		} else if (esource == maxmissingHButton) {
			szMessage = "     This parameter specifies the maximum number of missing values that are allowed for a gene.  "
					+ "A gene will be filtered if the number of time points for which there are no readings for the gene is "
					+ "greater than this parameter.  A gene will also be filtered if its expression value at the first time "
					+ "point is missing and log normalize data or normalize data was selected as the data transformation.";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == filterthresholdHButton) {
			szMessage = "     This parameter only applies if there is repeat data over different time periods.  "
					+ "In the case of one repeat data set, the correlation value of the gene's expression level "
					+ "between the original and repeat must have "
					+ "a correlation value above this parameter, otherwise the gene will be filtered.  "
					+ "In the case of multiple repeat sets, the mean pairwise correlation over all data sets "
					+ "must have a correlation value above this parameter, otherwise the gene will be filtered.";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == expressHButton) {

			szMessage = "  After transformation (Log normalize data, Normalize data, or No Normalization/add 0) "
					+ "if the absolute value of a gene's expression value at every time point "
					+ "is below this threshold, then the gene will be filtered.";

			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == samplepvalHButton) {
			szMessage = "This parameter specifies the number "
					+ "of random samples that should be made when computing multiple hypothesis "
					+ "corrected enrichment p-values by a randomization test.  A randomization test "
					+ "is used when the p-value enrichment is based on the actual size "
					+ "of the set of genes and 'Randomization' is selected next to the "
					+ "'Multiple hypothesis correction method for actual sized based enrichment' label. "
					+ "The Bonferroni correction is always used when the p-value enrichment is based on the "
					+ "expected size of the set of genes.  "
					+ "Increasing this parameter will lead to more accurate corrected p-values "
					+ "for the randomization test, but will also lead to longer execution time to compute the values.";

			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == mingoHButton) {
			szMessage = "For a category to be listed in a gene enrichment analysis table "
					+ "the number of genes in the set being analyzed that also belong to "
					+ "the category must be greater than or equal to this parameter.";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == mingolevelHButton) {
			szMessage = "   Any GO category whose level in the GO hierarchy is below this parameter will not be "
					+ "included in the GO analysis.  The categories Biological Process, Molecular Function, and "
					+ "Cellular Component are defined to be at level 1 in the hierarchy.  The level of any other "
					+ "term is the length of the longest path to one of these three GO terms in terms of the number "
					+ "of categories on the path.  This parameter thus allows one to exclude the most general GO categories.";

			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == methodHButton) {
			szMessage = "The significance level can be corrected for the fact that multiple profiles are being "
					+ "tested for significance.  The correction can be a Bonferroni correction where "
					+ "the significance level is divided by "
					+ "the number of model profiles or the less conservative False Discovery "
					+ "Rate control.  If 'none' is selected then no correction is made for the multiple "
					+ "significance tests.  Note that this parameter for multiple test "
					+ "correction for model profiles is unrelated to the corrected p-values in a GO enrichment analysis. ";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == xrefsourceHButton) {
			szMessage = "A cross reference file specifies that two or more symbols for the same gene are equivalent.  "
					+ "This file is optional, but is useful "
					+ "in the case where annotation of genes in the annotation file use a different naming "
					+ "convention than the genes in the data file. "
					+ "With a cross reference file it is possible "
					+ "to match a gene in the data file with its annotation in the annotation file "
					+ "even if the symbol used for the gene in the data file "
					+ "does not match the symbol used for the gene in the annotation "
					+ "file.\n\nUsing this menu a user can choose to "
					+ "either provide a cross reference file themselves ('User provided'), "
					+ "not to use a cross reference file ('No cross references'), or "
					+ "use an organism specific cross-reference file "
					+ "provided by the European Bioinformatics Institute (EBI) "
					+ "available from "
					+ "ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/.\n\nNote that leaving the cross-reference field blank "
					+ "under 'User Provided' is equivalent to selecting 'No cross references'.";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == downloadlistgoHButton) {
			szMessage = "If a box is checked then the corresponding file will be downloaded upon clicking execute "
					+ "and prior to analysis of the data. \n\n*The 'Annotation' field corresponds to the file listed in "
					+ "the 'Gene Annotation File' textbox.  Note that if the annotation source is user provided then "
					+ "the file cannot be downloaded automatically.  If the annotation file is not present locally it "
					+ "must be downloaded and thus the 'Annotation' check box will automatically be marked.  "
					+ "Gene annotation files are downloaded from "
					+ "ftp://ftp.geneontology.org/go/gene-associations/ unless it is "
					+ "an EBI data source in which case it will be downloaded from "
					+ "ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/.\n\n"
					+ "*The 'Cross References' file corresponds to the file under the 'Cross References File' text "
					+ "box, this field is only enabled if the Annotation Source is an EBI organism for which "
					+ "a cross reference file is provided.  If the cross reference file is not present locally "
					+ "on the computer it must be downloaded. Cross reference files are downloaded from "
					+ "ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/.\n\n"
					+ "*The 'Ontology' checkbox corresponds to the gene "
					+ "ontology specified in the gene_ontology.obo file.  It must be present if a non-user provided "
					+ "annotation source is selected, otherwise an attempt to download the file from "
					+ "http://www.geneontology.org/ontology/ will be made. \n\nCurrently:\n";
			szxrefval = xrefField.getText();
			szgoval = goField.getText();
			String szontologydate = "";
			String szgovaldate = "";
			String szxrefdate = "";

			File goannFile = new File(szgoval);
			if (goannFile.exists()) {
				szgovaldate = "The gene annotation file (" + szgoval
						+ ") was last updated "
						+ (new Date(goannFile.lastModified())).toString()
						+ ".\n";
			}

			File xrefFile = new File(szxrefval);
			if (xrefFile.exists()) {
				szxrefdate = "Gene annotation file (" + szxrefval
						+ ") was last updated "
						+ (new Date(xrefFile.lastModified())).toString()
						+ ".\n";
			}

			File oboFile = new File(szgocategoryval);
			if (oboFile.exists()) {
				szontologydate = "The gene ontology file (" + szgocategoryval
						+ ") was last updated "
						+ (new Date(oboFile.lastModified())).toString() + ".";
			}

			szMessage += szgovaldate + szxrefdate + szontologydate;
			Util.renderDialog(this, szMessage, 50, 100);
		} else if (esource == xrefHButton) {
			szMessage = "     A cross reference file specifies that two or more symbols for the same gene are equivalent.  "
					+ "This file is optional, but is useful "
					+ "in the case where annotation of genes in the annotation file use a different naming "
					+ "convention than the genes in the data file, "
					+ "and thus with a cross reference file it is possible "
					+ "to match a gene in the data file with its annotation in the annotation file "
					+ "even when the symbol used in the data file "
					+ "does not appear in the annotation file.\n\n     Note that the cross reference file is only used "
					+ "to map between gene symbols and not a spot ID and a gene symbol.  Any symbols on the same line are"
					+ " considered to be equivalent where symbols are delimited by tabs, a pipe(|), a comma (',')"
					+ " or a semicolon (;).  The file can either be in plain text or "
					+ "gzipped version of a text file in this format. \n\n"
					+ "Sample cross reference file:\n"
					+ "GeneA	SymbolA\n"
					+ "GeneB	SymbolB\n" + "GeneC	SymbolC";

			Util.renderDialog(this, szMessage, 50, 100);
		} else if (esource == presetsHButton) {
			szMessage = "The user can either select to provide their own gene annotation file ('User provided'), "
					+ "'No annotations', or use one of 35 gene annotation files available from the Gene Ontology "
					+ "Consortium.  More information about these 35 annotation files can be found here "
					+ "http://www.geneontology.org/GO.current.annotations.shtml and in the case of "
					+ "annotations from the European Bioinformatics Institue (EBI) also here http://www.ebi.ac.uk/GOA/.  "
					+ "If one of the predefined annotation files "
					+ "is selected then the annotation field will automatically be filled in and the option will be "
					+ "available to download the file.  If a user selects either "
					+ "the Arabidopsis, Chicken, Human, Mouse, Rat, or Zebrafish annotations "
					+ "from the European Bioinformatics Institute (EBI) "
					+ "then the cross-reference field will default to the cross-reference file corresponding to that "
					+ "annotation file.";
			Util.renderDialog(this, szMessage, 50, 100);
		} else if (esource == executeHButton) {
			szMessage = "Pressing the execute button causes any of the files that are checked next to "
					+ "'Download the latest' to be "
					+ "downloaded.   If the data file has two or more time points "
					+ "then the DREM algorithm will execute.  "
					+ "When the "
					+ "algorithm complete a new interface will appear showing an annotated dynamic map.";
			Util.renderDialog(this, szMessage, 50, 100);
		} else if (esource == miRNAComboBoxHButton) {
			szMessage = "The user can either provide their own miRNA-gene interaction file or use one of the five "
					+ "files that has been parsed from Computational Biology Center at Memorial Sloan-Kettering Cancer "
					+ "Center data.  The conserved data uses a cutoff of -0.1.  More information about the data in the"
					+ " files can be found at http://www.microrna.org/.";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == staticmiRNAInterHButton) {
			szMessage = "File encodes the static micro rna-gene interaction predictions as "
					+ "input to DREM.  The file is a tab delmited text file, that can be in one of "
					+ "two formats.  Either in a grid format or in a three column format.\n\n"
					+ "The grid format file is as follows:\n"
					+ "The first column of the file contains the gene identifier symbols.  "
					+ "The first row contains the micro rna identifiers.  "
					+ "The first entry of the first row is a label for the gene symbol column.  "
					+ "Each remaining entry corresponds to the relationship between a transcription factor and a "
					+ "gene.  Under the encoding an entry is greater than 0 if the transcription factor is predicted "
					+ "to regulate the gene and 0 otherwise.  If a three way encoding is used an entry is greater than 0 if "
					+ "the transcription factor is predicted to activate the gene, less than zero if it is predicted to repress "
					+ "the gene and 0 otherwise. Below is portion of a sample micro rna-gene interaction input file\n\n"
					+ "ID	ADR1	ARG80	ARG81	ARO80	BAS1	CAD1	CBF1\n"
					+ "YAL053W	0	0	0	0	0	0	1\n"
					+ "YAL054C	0	0	0	0	0	0	0.78\n"
					+ "YAL055W	0	0	0	0	0.5	0	0\n\n"
					+ "In the three column format the first column contains "
					+ "the micro rna, the second column "
					+ "the regulated gene, and the third column input value.  The first row is a header row "
					+ "where the header of the first column must be 'MIRNA' column, "
					+ "and the second column must have the header 'Gene'.  "
					+ "If a miRNA-gene pair is not present "
					+ "the input value is assumed to be 0.  When there are a lot of miRNAs and genes with a sparse number of "
					+ "non-zero entries then the three column format can lead to significant savings in space.\n"
					+ "MIRNA	Gene	Input\n"
					+ "BAS1	YAL055W	1\n"
					+ "CBF1	YAL053W	1\n" + "CBF1	YAL054C	1\n";
			;
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == staticmiRNAExpHButton) {
			szMessage = "This entry specifies a file that contains miRNA expression data.  "
					+ "A data file includes miRNA symbols and data values. "
					+ "The same gene symbol can appear multiple times in the data file.\n\n"
					+ "The file has the following "
					+ "formatting restrictions:\n"
					+ "* The first line contains the column headers delimited by tabs.\n"
					+ "* The remaining lines contain  "
					+ "gene symbols and then data delimited by tabs.\n"
					+ "* The first column contains the miRNA symbols.\n"
					+ "* In the gene field there can be multiple symbols listed delimited by "
					+ "either a pipe ('|'), comma (','), or a semi-colon (';').\n"
					+ "* The remaining columns contain the expression values in sequential order of time for the gene.\n"
					+ "* If a value is missing between two time points then the field should be left empty giving "
					+ "two tabs between the non-missing values.\n\n"
					+ "SAMPLE FILE:\n"
					+ "miRNA	0h	1h	3h	6h	12h\n"
					+ "YAL053W	-0.027	0.158	0.169	0.193	-0.165\n"
					+ "YAL054C	0.183	-0.068	-0.134	-0.252	0.177\n"
					+ "YAL055W	-0.923	-0.51	-0.718	-0.512	-0.668";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == miRNARepeatHButton) {
			szMessage = "     Clicking on the 'Repeat Data' button brings up a dialog box "
					+ "to specify repeat data files of the experiment.  Repeat data files "
					+ "are optional.  The format of repeat data is the same as the original data.  "
					+ "If included the repeat data values will be averaged using the median with the "
					+ "original experiment values.  "
					+ "Repeat data can either represent repeat measurements taken concurrently with the "
					+ "original experiment, or distinct full repeat experiments taken at different time periods.  "
					+ "If the data is of the latter type genes will be filtered that do not display a consistent "
					+ "expression profile between repeats based on the 'Minimum Correlation between Repeats' "
					+ "parameter under the 'Filtering' panel on the options menu.  "
					+ "If the button is yellow then there is currently repeat data loaded, otherwise the button is gray.";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == miRNANormHButton) {
			szMessage = "All time series will be transformed so that the "
					+ "time series starts at 0.  "
					+ "This can be done in one of three ways based on the option selected to the left.  "
					+ "Given a time series vector of values for a miRNA (v_0,v_1,...,v_n) the options are:\n"
					+ "1.  'Log normalize data'  \u2212  the vector "
					+ "will be transformed to (0,log\u2082(v_1)\u2212log\u2082(v_0),...,log\u2082(v_n)\u2212log\u2082(v_(n-1))).  "
					+ "Note that any values which are 0 or negative will be treated as missing.\n"
					+ "2.  'Normalize data'  \u2212  the vector "
					+ "will be transformed to (0,v_1\u2212v_0,...,v_n\u2212v_(n-1))\n"
					+ "3.  'No normalization/add 0'  \u2212  a 0 will be inserted transforming the vector to "
					+ "(0,v_0,v_1,...,v_n)\n\n"
					+ "*If the data is not already in log space as often is the case if it is "
					+ "from an Oligonucleotide array, then the "
					+ "'Log normalize data' should be selected.\n"
					+ "*If the data is already in log space as is often the case if the data is "
					+ "from a two channel cDNA array and "
					+ "a time point 0 experiment was "
					+ "conducted, then the 'Normalize data' option should be selected.\n"
					+ "*If the data is already in log space and no time point 0 experiment was "
					+ "conducted, then the 'No normalization/add 0' option should be selected.";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == miRNAScoringHButton) {
			szMessage = "The user can elect to use activity scores for "
					+ "transcription factors. This will scale the regulator interaction values "
					+ "by its expression. By default TF expression scaling is off  ";
			//szMessage = "The user can elect to use activity scores for miRNA and/or "
			//		+ "transcription factors.  This will scale the regulator interaction values "
			//		+ "by its expression.  By default miRNA will have their activity scaled and "
			//		+ "transcription factors will not.";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == miRNAFilterHButton) {
			szMessage = "If this box is checked then miRNA with no expression will be "
					+ "filtered from the miRNA interaction data.";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == miRNAWeightHButton) {
			szMessage = "This number will be used to multiplicatively scale the activity "
					+ "value used to compute the activity score.";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == miRNATFWeightHButton) {
			szMessage = "This specifies the minimum magnitude of transcription factor "
				+ "regulation after expression scaling.  This feature allows to keep "
				+ "TFs that operate at low expression levels or that are post-transcriptionally "
				+ "activated. ";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == fastaDataHButton) {
			szMessage = "This specifies the file used to map genes to their associated sequence"
					+ " in FASTA format. "
					+ "The gene names should be on a line starting with a > symbol and followed "
					+ "by the fasta formatted gene.  Lines with a # character will be ignored. "
					+ "For example:\n"
					+ "#Comment\n"
					+ ">NOC2L\n"
					+ "GTTCGTGGCGCTGGCCCGGTCTCCGCGGATCGGAGGCGAAGCCAGCCTGGCCCTCGGGTC\n"
					+ "GCCCGTCTTTTGTCGAGCTGGCGACATCAGTGCGGTTCCCACTGCCCCTCGCGTTTTCCA\n"
					+ ">PLEKHN1\n"
					+ "TGGCCGTCTAGGGGGCATGTGGCCTCCCTGAGTCCCCTTAAGCCTTGGGGACCCTGACTC\n"
					+ "GGGTCTGTGGCGAGGGGGCCCAGGCAGGAGGGGAGGCTGCGGTGGCTTTGGCCGCCGTCT";
	
			
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == decodPathHButton) {
			szMessage = "This specifies the DECOD file to optionally run at a split.";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		} else if (esource == regScoreHButton) {
			szMessage = "This file specifies the regulator scores.  These are used"
					+ " to multiplicatively scale the regulators' binding interactions.";
			Util.renderDialog(theOptions, szMessage, 50, 100);
		}
	}

	/**
	 * Places szMessage in thedialog window with the title 'Help'
	 */
	public static void renderDialog(JDialog thedialog, String szMessage,
			int noffsetx, int noffsety) {
		renderDialog(thedialog, szMessage, noffsetx, noffsety, "Help");
	}

	/**
	 * Places szMessage in thedialog window with the title szTitle
	 */
	public static void renderDialog(JDialog thedialog, String szMessage,
			int noffsetx, int noffsety, String szTitle) {
		final JDialog thedialogf = thedialog;
		final JTextArea textAreaf = new JTextArea(szMessage);
		final int noffsetxf = noffsetx;
		final int noffsetyf = noffsety;
		final String szTitlef = szTitle;
		final int nlengthf = szMessage.length();
		javax.swing.SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				JDialog helpDialog = new JDialog(thedialogf, szTitlef, false);
				Container theHelpDialogPane = helpDialog.getContentPane();

				helpDialog.setBackground(Color.white);
				theHelpDialogPane.setBackground(Color.white);

				textAreaf.setLineWrap(true);
				textAreaf.setWrapStyleWord(true);

				textAreaf.setBackground(Color.white);
				textAreaf.setEditable(false);
				JScrollPane jsp = new JScrollPane(textAreaf);
				theHelpDialogPane.add(jsp);

				helpDialog.setLocation(thedialogf.getX() + noffsetxf,
						thedialogf.getY() + noffsetyf);
				if (nlengthf < 600) {
					helpDialog.setSize(700, 150);
				} else if (nlengthf < 1000) {
					helpDialog.setSize(700, 250);
				} else {
					helpDialog.setSize(700, 350);
				}

				helpDialog.setDefaultCloseOperation(DISPOSE_ON_CLOSE);
				helpDialog.setVisible(true);
			}
		});
	}

	// ////////////////////////////////////////////////////
	/**
	 * Renders the tabbed set of option window panes
	 */
	public void makeOptionsDialog() {
		theOptions = new JDialog(this, "Options", true);

		Container theDialogContainer = theOptions.getContentPane();

		theDialogContainer.setBackground(Color.white);
		JTabbedPane tabbedPane = new JTabbedPane();
                //
                JComponent panelMethy=makeMethyPanel();
                panelMethy.setBackground(lightBlue);
                tabbedPane.addTab("Epigenomics Option",null,panelMethy,"Options pertaining to Epigenomic data");
                
                //
                JComponent panelProteomics = makeProteomicsPanel();
                panelProteomics.setBackground(lightBlue);
                tabbedPane.addTab("Proteomics Option",null,panelProteomics,
                        "Options pertaining to proteomics data");
                
                //
		JComponent panelFiltering = makeFilterPanel();
		panelFiltering.setBackground(lightBlue);
		tabbedPane.addTab("Filtering Options", null, panelFiltering,
				"Options pertaining to gene filtering");

		JComponent panelInit = makeSearchPanel();
		panelInit.setBackground(lightBlue);
		tabbedPane.addTab("Search Options", null, panelInit,
				"Options pertaining to searching for the best model");

		JComponent paneladdsearchInit = makeAdditionalSearchPanel();
		panelInit.setBackground(lightBlue);
		tabbedPane.addTab("Model Selection Options", null, paneladdsearchInit,
				"Options pertaining to model selection");

		JComponent panelAnnotation = makeAnnotationPanel();
		panelAnnotation.setBackground(lightBlue);
		tabbedPane.addTab("Gene Annotations", null, panelAnnotation,
				"Options pertaining to gene annotations");

		JComponent panelExecution = makeGOAnalysisPanel();
		panelExecution.setBackground(lightBlue);
		tabbedPane.addTab("GO Analysis", null, panelExecution,
				"Options pertaining to Gene Ontology enrichment analysis");
		theDialogContainer.add(tabbedPane);
		
		//TODO: add back in sdrem tab
		JComponent panelSDREM = makeSDREMPanel();
		panelSDREM.setBackground(lightBlue);
		//tabbedPane.addTab("SDREM Options", null, panelSDREM,
		//		"Options pertaining to the use of SDREM.");
		
		JComponent panelDECOD = makeDECODPanel();
		panelDECOD.setBackground(lightBlue);
		tabbedPane
				.addTab("DECOD Options", null, panelDECOD,
						"Options pertaining to using DECOD at splits of the final model");
		
		JComponent panelScale = makeExpressionScalingPanel();
		panelScale.setBackground(lightBlue);
		tabbedPane.addTab("Expression Scaling Options", null, panelScale,
				"Options pertaining to scaling regulator binding data based "
					+ "on regulator expression");
		
		//TODO: add backin mirna panel
		JComponent panelMIRNA = makeMIRNAPanel();
		panelMIRNA.setBackground(lightBlue);
		tabbedPane.addTab("microRNA Option", null, panelMIRNA,
				"Options pertaining to include miRNA target predictions"
						+ " into model learning");
		
		theOptions.pack();

		theOptions.setDefaultCloseOperation(DO_NOTHING_ON_CLOSE);
		theOptions.addWindowListener(new WindowAdapter() {
			public void windowClosing(WindowEvent we) {
				boolean bok = true;
				try {
					thespinnerepsilon.commitEdit();
					thespinnerprunepath.commitEdit();
					thespinnerdelaypath.commitEdit();
					thespinnermergepath.commitEdit();
					thespinnernodepenalty.commitEdit();
					thespinnerconvergence.commitEdit();
					thespinnerminstddev.commitEdit();
					thespinnerepsilondiff.commitEdit();
					thespinnerprunepathdiff.commitEdit();
					thespinnerdelaypathdiff.commitEdit();
					thespinnermergepathdiff.commitEdit();
					thespinnernumchild.commitEdit();
					spinnerMIRNAWeight.commitEdit();
					spinnerTFWeight.commitEdit();
					spinnerdProbBind.commitEdit();

					thespinnerseed.commitEdit();
					thespinnermingo.commitEdit();
					thespinnermingolevel.commitEdit();
					thespinnermaxmissing.commitEdit();
					thespinnerfilterthreshold.commitEdit();
					thespinnerexpress.commitEdit();
					thespinnersamplepval.commitEdit();
				} catch (ParseException ex) {
					Toolkit.getDefaultToolkit().beep();
					bok = false;
				}

				if (bok) {
					theOptions.setVisible(false);
				}
			}
		});
	}
        
        // /////////////////////////////////////////////////
        /**
         * Makes the Methylation panel Option 
         * 
         * */
        protected JComponent makeMethyPanel(){
            
            MethyFileButton.addActionListener(this);
            MethyHButton.addActionListener(this);
            MethyGTFButton.addActionListener(this);
            MethyGTFHButton.addActionListener(this);
           
            
            MethyLabel=new JLabel("Epigenomic data File:");
            MethyField= new JTextField(djMethyFile, JLabel.TRAILING);
            //File Opener for Methylation data
            JPanel mFileOpener = new JPanel(new SpringLayout());
            mFileOpener.add(MethyLabel);
            mFileOpener.add(MethyField);
            mFileOpener.add(MethyFileButton);
            mFileOpener.add(MethyHButton); 
            SpringUtilities.makeCompactGrid(mFileOpener,1,4,3,3,3,3);
            mFileOpener.setBackground(lightBlue);
            MethyField.setColumns(40);
            int nwidth = (int) MethyField.getPreferredSize().getWidth();
            MethyFileButton.setMaximumSize( MethyFileButton.getPreferredSize() );
            int nheight = (int) MethyFileButton.getPreferredSize().getHeight();
            MethyField.setPreferredSize(new Dimension(nwidth, nheight));
            
            // File Opener for Methylation TF Motif
            MethyGTFLabel=new JLabel("GTF file:");
            MethyGTFField=new JTextField (djMethyGTF);
            
            JPanel motifFileOpener =new JPanel (new SpringLayout());
            motifFileOpener.add (MethyGTFLabel);
            motifFileOpener.add (MethyGTFField);
            motifFileOpener.add (MethyGTFButton);
            motifFileOpener.add (MethyGTFHButton);
            SpringUtilities.makeCompactGrid(motifFileOpener,1,4,3,3,3,3);
            motifFileOpener.setBackground(lightBlue);
            
            //add all to the panel 
           
            JPanel MPanel=new JPanel(new SpringLayout());
            MPanel.add(mFileOpener);
            MPanel.add(motifFileOpener);
          
            MPanel.add(Box.createRigidArea(new Dimension(0,210)));
            MPanel.setBackground(lightBlue);
            SpringUtilities.makeCompactGrid(MPanel,3,1,3,3,3,3);
                
            return MPanel;
        }
        
        
        // ///////////////////////////////////////////////
       /**
        * Makes the Proteomics Panel Option 
        * 
        */
        //FIXME Make Proteomics Panel
        protected JComponent makeProteomicsPanel(){
            // Listerner section 
            JFileChooser PChooser=new JFileChooser(); 
            ProteFileButton.addActionListener(this);
            ProteRepeatButton.addActionListener(this);
            ProteHButton.addActionListener(this);
            ProteHCButton.addActionListener(this);
            ProteHRButton.addActionListener(this);
            ProtePPIButton.addActionListener(this);
            ProteHPPIButton.addActionListener(this);
           
             
            ProteTFCheckBox = new JCheckBox("Only Use Proteomcis Data for TFs",bProteTF);
            ProteTFCheckBox.setBackground(lightBlue);
            ProteAllCheckBox = new JCheckBox("Use Proteomics data for all Proteins",bProteAll);
            ProteAllCheckBox.setBackground(lightBlue);
            ProteNoCheckBox = new JCheckBox("Do Not Use Proteomics data",bProteNo);
            ProteNoCheckBox.setBackground(lightBlue);
            ProteNoCheckBox.addChangeListener(this);
            
            
            
            //proteomics file dialog
            ProteFileLabel = new JLabel("Proteomics Data File:");
            ProteField = new JTextField(djProteFile, JLabel.TRAILING);
            ProteField.setColumns(40);
            int nwidth = (int) ProteField.getPreferredSize().getWidth();
            ProteFileButton.setMaximumSize( ProteFileButton.getPreferredSize() );
            int nheight = (int) ProteFileButton.getPreferredSize().getHeight();
            
            ProteField.setPreferredSize(new Dimension(nwidth, nheight));
            
            JPanel pFileOpener = new JPanel(new SpringLayout());
            pFileOpener.add(ProteFileLabel);
            pFileOpener.add(ProteField);
            pFileOpener.add(ProteFileButton);
            pFileOpener.add(ProteHButton);
            SpringUtilities.makeCompactGrid(pFileOpener,1,4,3,3,3,3);
            pFileOpener.setBackground(lightBlue);
           
            //prote checkbox
            ButtonGroup pCheckBoxGroup =new ButtonGroup();
            pCheckBoxGroup.add(ProteTFCheckBox);
            pCheckBoxGroup.add(ProteAllCheckBox);
            pCheckBoxGroup.add(ProteNoCheckBox);
            
            JPanel pCheckBox = new JPanel(new SpringLayout());   
            pCheckBox.add(ProteTFCheckBox);
            pCheckBox.add(ProteAllCheckBox);
            pCheckBox.add(ProteNoCheckBox);
            pCheckBox.add(ProteHCButton);
            SpringUtilities.makeCompactGrid(pCheckBox,1,4,3,3,3,3);
            pCheckBox.setBackground(lightBlue);
            
            //repeats
            JPanel repeatOpener =new JPanel (new SpringLayout());
            ButtonGroup ProteNormgroup =new ButtonGroup();
            ProtelognormButton = new JRadioButton("Log normalize data");
            ProtelognormButton.setBackground(lightBlue);
            ProtenormButton = new JRadioButton("Normalize data");
            ProtenormButton.setBackground(lightBlue);
            ProtenonormButton = new JRadioButton("No normalization/add 0");
            ProtenonormButton.setBackground(lightBlue);
            ProteNormgroup.add(ProtelognormButton);
            ProteNormgroup.add(ProtenormButton);
            ProteNormgroup.add(ProtenonormButton);
            if (nnormalizeDEF == 0) {
                ProtelognormButton.setSelected(true);
            } else if (nnormalizeDEF == 1) {
                ProtenormButton.setSelected(true);
            } else {
                ProtenonormButton.setSelected(true);
            }
            repeatOpener.add(ProteRepeatButton);
            repeatOpener.add(ProtelognormButton);
            repeatOpener.add(ProtenormButton);
            repeatOpener.add(ProtenonormButton);
            repeatOpener.add(ProteHRButton);
            SpringUtilities.makeCompactGrid(repeatOpener,1,5,3,3,3,3);
            repeatOpener.setBackground(lightBlue);
            
            //PPI
            ProtePPILabel = new JLabel("Protein-Protein Interaction File:");
            ProtePPIField = new JTextField(djPPIFile, JLabel.TRAILING);
            ProtePPIField.setColumns(40);
            ProtePPIButton.setMaximumSize( ProteFileButton.getPreferredSize() );
            
            ProtePPIField.setPreferredSize(new Dimension(nwidth, nheight));
            
            JPanel pPPIOpener = new JPanel(new SpringLayout());
            pPPIOpener.add(ProtePPILabel);
            pPPIOpener.add(ProtePPIField);
            pPPIOpener.add(ProtePPIButton);
            pPPIOpener.add(ProteHPPIButton);
            SpringUtilities.makeCompactGrid(pPPIOpener,1,4,3,3,3,3);
            pPPIOpener.setBackground(lightBlue);
           
            
            //add all to the panel 
           
            JPanel PPanel=new JPanel(new SpringLayout());
            PPanel.add(pCheckBox);
            PPanel.add(pFileOpener);
            PPanel.add(repeatOpener);
            PPanel.add(pPPIOpener);
            PPanel.add(Box.createRigidArea(new Dimension(0, 150)));
            PPanel.setBackground(lightBlue);
            
            SpringUtilities.makeCompactGrid(PPanel,5,1,3,3,3,3);
                
            toggleUseProteEnabled(!(ProteNoCheckBox.isSelected()));
            return PPanel;
        }
        /**
	 * Toggles which model selection variables are enabled based on the value of
	 * buseProte
	 */
	private void toggleUseProteEnabled(boolean buseProte) {
                ProteFileLabel.setEnabled(buseProte);
                ProteField.setEnabled(buseProte);
                ProteFileButton.setEnabled(buseProte);
                ProteHButton.setEnabled(buseProte);
                ProteRepeatButton.setEnabled(buseProte);
                ProtelognormButton.setEnabled(buseProte);
                ProtenormButton.setEnabled(buseProte);
                ProtenonormButton.setEnabled(buseProte);
                ProteHRButton.setEnabled(buseProte);
                ProteHPPIButton.setEnabled(buseProte);
                ProtePPIButton.setEnabled(buseProte);
	}
        
        
	// /////////////////////////////////////////////
	/**
	 * Makes the GO annotation options panel
	 */
	protected JComponent makeAnnotationPanel() {

		evidenceHButton.addActionListener(this);
		taxonHButton.addActionListener(this);
		ontoHButton.addActionListener(this);

		pcheck = new JCheckBox("Biological Process", bpontoDEF);
		fcheck = new JCheckBox("Molecular Function", bfontoDEF);
		ccheck = new JCheckBox("Cellular Component", bcontoDEF);

		JPanel ponto = new JPanel();
		JLabel ontoLabel = new JLabel("Only include annotations of type: ",
				JLabel.TRAILING);
		ponto.add(ontoLabel);
		ponto.setBackground(lightBlue);
		pcheck.setBackground(lightBlue);
		fcheck.setBackground(lightBlue);
		ccheck.setBackground(lightBlue);
		ponto.add(pcheck);
		ponto.add(fcheck);
		ponto.add(ccheck);
		ponto.add(ontoHButton);

		JPanel pevidence = new JPanel(new SpringLayout());
		JLabel evidenceLabel = new JLabel(
				"Exclude annotations with these evidence codes:",
				JLabel.TRAILING);
		evidenceField = new JTextField(szevidenceDEF, JLabel.TRAILING);
		evidenceField.setColumns(20);

		evidenceLabel.setLabelFor(evidenceField);

		JLabel taxonLabel = new JLabel(
				"Only include annotations with these taxon IDs:",
				JLabel.TRAILING);
		taxonField = new JTextField(sztaxonDEF, JLabel.TRAILING);
		taxonField.setColumns(20);

		pevidence.add(taxonLabel);
		pevidence.add(taxonField);
		taxonLabel.setLabelFor(taxonField);
		pevidence.add(taxonHButton);

		pevidence.add(evidenceLabel);
		pevidence.add(evidenceField);
		pevidence.setBackground(lightBlue);
		pevidence.add(evidenceHButton);

		JPanel p2 = new JPanel();
		p2.add(categoryIDLabel);
		int nwidth = (int) orig1Field.getPreferredSize().getWidth();
		int nheight = (int) categoryIDButton.getPreferredSize().getHeight();
		taxonField.setPreferredSize(new Dimension(nwidth, nheight));
		evidenceField.setPreferredSize(new Dimension(nwidth, nheight));
		categoryIDField.setPreferredSize(new Dimension(nwidth, nheight));
		SpringUtilities.makeCompactGrid(pevidence, 2, 3, 4, 5, 6, 6);
		categoryIDLabel.setLabelFor(categoryIDField);
		p2.add(categoryIDField);
		p2.add(categoryIDButton);
		p2.add(categoryIDHButton);
		p2.setBackground(lightBlue);
		categoryIDButton.addActionListener(this);
		categoryIDHButton.addActionListener(this);

		JPanel entirepanel = new JPanel();
		BoxLayout layout = new BoxLayout(entirepanel, BoxLayout.Y_AXIS);
		entirepanel.setLayout(layout);
		p2.setBackground(lightBlue);

		entirepanel.add(ponto);
		entirepanel.add(pevidence);

		entirepanel.add(p2);
		entirepanel.add(Box.createRigidArea(new Dimension(0, 90)));

		entirepanel.setLayout(layout);
		entirepanel.setBackground(lightBlue);

		return entirepanel;
	}

	// /////////////////////////////////////////////
	/**
	 * Makes the GO analysis options panel
	 */
	protected JComponent makeGOAnalysisPanel() {
		JPanel p = new JPanel(new SpringLayout());

		p.add(mingolevelLabel);
		p.add(thespinnermingolevel);
		p.add(mingolevelHButton);
		mingolevelHButton.addActionListener(this);

		p.add(mingoLabel);
		p.add(thespinnermingo);
		p.add(mingoHButton);
		mingoHButton.addActionListener(this);

		p.add(samplepvalLabel);
		p.add(thespinnersamplepval);
		p.add(samplepvalHButton);
		samplepvalHButton.addActionListener(this);

		brandomgoval = brandomgoDEF;
		JPanel correctPanel = new JPanel();
		randomgoLabel = new JLabel(szrandomgo);
		correctPanel.add(randomgoLabel);
		bfgoButton = new JRadioButton("Bonferroni");
		bfgoButton.setBackground(lightBlue);
		randomgoButton = new JRadioButton("Randomization");
		randomgoButton.setBackground(lightBlue);

		ButtonGroup group = new ButtonGroup();
		if (brandomgoval) {
			randomgoButton.setSelected(true);
		} else {
			bfgoButton.setSelected(true);
		}
		p.setBackground(lightBlue);
		group.add(bfgoButton);
		group.add(randomgoButton);
		correctPanel.add(bfgoButton);
		correctPanel.add(randomgoButton);
		correctPanel.add(randomgoHButton);
		correctPanel.setBackground(lightBlue);

		randomgoHButton.addActionListener(this);
		SpringUtilities.makeCompactGrid(p, 3, 3, 5, 5, 5, 5);

		JPanel entirepanel = new JPanel();
		BoxLayout layout = new BoxLayout(entirepanel, BoxLayout.Y_AXIS);
		entirepanel.setLayout(layout);
		entirepanel.add(p);
		entirepanel.add(correctPanel);
		entirepanel.add(Box.createRigidArea(new Dimension(0, 50)));
		entirepanel.setLayout(layout);
		entirepanel.setBackground(lightBlue);

		return entirepanel;
	}

	// //////////////////////////////////////////////////////////////////////////////////////

	/**
	 * Makes the panel showing the model selection options
	 */
	protected JComponent makeAdditionalSearchPanel() {

		JPanel p = new JPanel();
		p.setBackground(lightBlue);

		BoxLayout layout = new BoxLayout(p, BoxLayout.Y_AXIS);
		p.setLayout(layout);

		JPanel modelPanel = new JPanel();
		modelPanel.setBackground(lightBlue);
		p.add(modelPanel);
		penalizedButton = new JRadioButton("Penalized Likelihood");
		penalizedButton.setBackground(lightBlue);
		traintestButton = new JRadioButton("Train-Test");
		traintestButton.setBackground(lightBlue);
		modelLabel = new JLabel("Model Selection Framework:");
		modelframeworkHButton.addActionListener(this);
		modelPanel.add(modelLabel);
		modelPanel.add(penalizedButton);
		modelPanel.add(traintestButton);
		modelPanel.add(modelframeworkHButton);
		modelGroup.add(penalizedButton);
		modelGroup.add(traintestButton);

		if (bPENALIZEDDEF) {
			penalizedButton.setSelected(true);
		} else {
			traintestButton.setSelected(true);
		}

		traintestButton.addChangeListener(this);
                
                //
		SpinnerNumberModel snnodepenalty = new SpinnerNumberModel(new Double(
				dNODEPENALTYDEF), new Double(0), null, new Double(5));
		thespinnernodepenalty = new JSpinner(snnodepenalty);
		thespinnernodepenalty.setPreferredSize(new Dimension(50, 20));
		thespinnernodepenalty.setMinimumSize(new Dimension(50, 20));
		thespinnernodepenalty.setMaximumSize(new Dimension(50, 20));
               
		JPanel pnodepenaltypanel = new JPanel(new SpringLayout());
		nodepenaltyLabel = new JLabel(sznodepenalty, JLabel.TRAILING);
		pnodepenaltypanel.setBackground(lightBlue);
		pnodepenaltypanel.add(nodepenaltyLabel);
		pnodepenaltypanel.add(thespinnernodepenalty);
		nodepenaltyHButton.addActionListener(this);
		pnodepenaltypanel.add(nodepenaltyHButton);
		p.add(pnodepenaltypanel);
                
		SpringUtilities.makeCompactGrid(pnodepenaltypanel, 1, 3, 5, 5, 5, 5);
		
                
                
                //TODO: Add proteomics weight panel
                SpinnerNumberModel proteWeight = new SpinnerNumberModel(new Double(1),new Double(0),null,0.1);
                theproteWeight = new JSpinner(proteWeight);
                theproteWeight.setPreferredSize(new Dimension(50, 20));
		theproteWeight.setMinimumSize(new Dimension(50, 20));
		theproteWeight.setMaximumSize(new Dimension(50, 20));

                JPanel proteweightpanel = new JPanel(new SpringLayout());
                pproteweightLabel=new JLabel("Relative weight of proteomics data against to epigenomics data", JLabel.TRAILING);
                proteweightpanel.setBackground(lightBlue);
                proteweightpanel.add(pproteweightLabel);
                proteweightpanel.add(theproteWeight); // proteomics data weight spinner
                proteweightHButton.addActionListener(this);
                proteweightpanel.add(proteweightHButton);
                SpringUtilities.makeCompactGrid(proteweightpanel,1,3,5,5,5,5);
                p.add(proteweightpanel);
                
               // TODO: ADD epigenetics weight panel
               /*
                SpinnerNumberModel epigeneticWeight = new SpinnerNumberModel(new Double(1),new Double(0),null,0.1);
                theepigeneticWeight = new JSpinner(epigeneticWeight);
                theepigeneticWeight.setPreferredSize(new Dimension(50, 20));
		theepigeneticWeight.setMinimumSize(new Dimension(50, 20));
		theepigeneticWeight.setMaximumSize(new Dimension(50, 20));

                JPanel epigeneticweightpanel = new JPanel(new SpringLayout());
                pepigeneticWeightLabel=new JLabel("Epigenomics data weight", JLabel.TRAILING);
                epigeneticweightpanel.setBackground(lightBlue);
                epigeneticweightpanel.add(pepigeneticWeightLabel);
                epigeneticweightpanel.add(theepigeneticWeight); // proteomics data weight spinner
                epigeneticweightHButton.addActionListener(this);
                epigeneticweightpanel.add(epigeneticweightHButton);
                SpringUtilities.makeCompactGrid(epigeneticweightpanel,1,3,5,5,5,5);
                p.add(epigeneticweightpanel);
                */
                
                //
                JPanel pspinnerseedpanel = new JPanel(new SpringLayout());
		seedLabel = new JLabel(szseed, JLabel.TRAILING);
		pspinnerseedpanel.add(seedLabel);
		pspinnerseedpanel.add(thespinnerseed);
		pspinnerseedpanel.add(seedHButton);
		pspinnerseedpanel.setBackground(lightBlue);
		seedHButton.addActionListener(this);
		p.add(pspinnerseedpanel);

		SpringUtilities.makeCompactGrid(pspinnerseedpanel, 1, 3, 5, 5, 5, 5);
		JPanel pspinnerpanel = new JPanel(new SpringLayout());
		pspinnerpanel.setBackground(lightBlue);

		SpinnerNumberModel snepsilon = new SpinnerNumberModel(new Double(
				dMinScoreDEF), new Double(0), null, new Double(.01));
		thespinnerepsilon = new JSpinner(snepsilon);
		thespinnerepsilon.setPreferredSize(new Dimension(50, 20));
		thespinnerepsilon.setMinimumSize(new Dimension(50, 20));
		thespinnerepsilon.setMaximumSize(new Dimension(50, 20));

		SpinnerNumberModel snprunepath = new SpinnerNumberModel(new Double(
				dPRUNEPATHDEF), new Double(0), null, new Double(.01));
		thespinnerprunepath = new JSpinner(snprunepath);
		thespinnerprunepath.setPreferredSize(new Dimension(50, 20));
		thespinnerprunepath.setMinimumSize(new Dimension(50, 20));
		thespinnerprunepath.setMaximumSize(new Dimension(50, 20));

		SpinnerNumberModel sndelaypath = new SpinnerNumberModel(new Double(
				dDELAYPATHDEF), new Double(0), null, new Double(.01));
		thespinnerdelaypath = new JSpinner(sndelaypath);
		thespinnerdelaypath.setPreferredSize(new Dimension(50, 20));
		thespinnerdelaypath.setMinimumSize(new Dimension(50, 20));
		thespinnerdelaypath.setMaximumSize(new Dimension(50, 20));

		SpinnerNumberModel snmergepath = new SpinnerNumberModel(new Double(
				dDMERGEPATHDEF), new Double(0), null, new Double(.01));
		thespinnermergepath = new JSpinner(snmergepath);
		thespinnermergepath.setPreferredSize(new Dimension(50, 20));
		thespinnermergepath.setMinimumSize(new Dimension(50, 20));
		thespinnermergepath.setMaximumSize(new Dimension(50, 20));

		SpinnerNumberModel snepsilondiff = new SpinnerNumberModel(new Double(
				dMinScoreDIFFDEF), new Double(0), null, new Double(.01));
		thespinnerepsilondiff = new JSpinner(snepsilondiff);
		thespinnerepsilondiff.setPreferredSize(new Dimension(50, 20));
		thespinnerepsilondiff.setMinimumSize(new Dimension(50, 20));
		thespinnerepsilondiff.setMaximumSize(new Dimension(50, 20));

		SpinnerNumberModel snprunepathdiff = new SpinnerNumberModel(new Double(
				dPRUNEPATHDIFFDEF), null, new Double(0), new Double(.01));
		thespinnerprunepathdiff = new JSpinner(snprunepathdiff);
		thespinnerprunepathdiff.setPreferredSize(new Dimension(50, 20));
		thespinnerprunepathdiff.setMinimumSize(new Dimension(50, 20));
		thespinnerprunepathdiff.setMaximumSize(new Dimension(50, 20));

		SpinnerNumberModel sndelaypathdiff = new SpinnerNumberModel(new Double(
				dDELAYPATHDIFFDEF), null, new Double(0), new Double(.01));
		thespinnerdelaypathdiff = new JSpinner(sndelaypathdiff);
		thespinnerdelaypathdiff.setPreferredSize(new Dimension(50, 20));
		thespinnerdelaypathdiff.setMinimumSize(new Dimension(50, 20));
		thespinnerdelaypathdiff.setMaximumSize(new Dimension(50, 20));

		SpinnerNumberModel snmergepathdiff = new SpinnerNumberModel(new Double(
				dDMERGEPATHDIFFDEF), null, new Double(0), new Double(.01));
		thespinnermergepathdiff = new JSpinner(snmergepathdiff);
		thespinnermergepathdiff.setPreferredSize(new Dimension(50, 20));
		thespinnermergepathdiff.setMinimumSize(new Dimension(50, 20));
		thespinnermergepathdiff.setMaximumSize(new Dimension(50, 20));

		epsilonLabel = new JLabel(szepsilon, JLabel.TRAILING);
		epsilonLabeldiff = new JLabel(szepsilondiff, JLabel.TRAILING);
		pspinnerpanel.add(epsilonLabel);
		pspinnerpanel.add(thespinnerepsilon);
		pspinnerpanel.add(epsilonLabeldiff);
		pspinnerpanel.add(thespinnerepsilondiff);
		pspinnerpanel.add(epsilonHButton);
		pspinnerpanel.setBackground(lightBlue);
		epsilonHButton.addActionListener(this);

		prunepathLabel = new JLabel(szprunepath, JLabel.TRAILING);
		prunepathLabeldiff = new JLabel(szprunepathdiff, JLabel.TRAILING);
		pspinnerpanel.add(prunepathLabel);
		pspinnerpanel.add(thespinnerprunepath);
		pspinnerpanel.add(prunepathLabeldiff);
		pspinnerpanel.add(thespinnerprunepathdiff);
		pspinnerpanel.add(prunepathHButton);
		prunepathHButton.addActionListener(this);

		delaypathLabel = new JLabel(szdelaypath, JLabel.TRAILING);
		delaypathLabeldiff = new JLabel(szdelaypathdiff, JLabel.TRAILING);
		pspinnerpanel.add(delaypathLabel);
		pspinnerpanel.add(thespinnerdelaypath);
		pspinnerpanel.add(delaypathLabeldiff);
		pspinnerpanel.add(thespinnerdelaypathdiff);
		pspinnerpanel.add(delaypathHButton);
		delaypathHButton.addActionListener(this);

		mergeLabel = new JLabel(szmergepath, JLabel.TRAILING);
		mergeLabeldiff = new JLabel(szmergepathdiff, JLabel.TRAILING);
		pspinnerpanel.add(mergeLabel);
		pspinnerpanel.add(thespinnermergepath);
		pspinnerpanel.add(mergeLabeldiff);
		pspinnerpanel.add(thespinnermergepathdiff);
		pspinnerpanel.add(mergepathHButton);

		mergepathHButton.addActionListener(this);

		SpringUtilities.makeCompactGrid(pspinnerpanel, 4, 5, 5, 5, 5, 5);
		p.add(pspinnerpanel);

		toggleEnabled(traintestButton.isSelected());

		return p;
	}

	/**
	 * Toggles which model selection variables are enabled based on the value of
	 * btraintest
	 */
	private void toggleEnabled(boolean btraintest) {
		thespinnerdelaypath.setEnabled(btraintest);
		thespinnerdelaypathdiff.setEnabled(btraintest);
		thespinnermergepath.setEnabled(btraintest);
		thespinnermergepathdiff.setEnabled(btraintest);
		mergeLabeldiff.setEnabled(btraintest);
		mergeLabel.setEnabled(btraintest);
		delaypathLabeldiff.setEnabled(btraintest);
		delaypathLabel.setEnabled(btraintest);
		prunepathLabel.setEnabled(btraintest);
		prunepathLabeldiff.setEnabled(btraintest);
		thespinnerprunepath.setEnabled(btraintest);
		thespinnerprunepathdiff.setEnabled(btraintest);
		epsilonLabel.setEnabled(btraintest);
		epsilonLabeldiff.setEnabled(btraintest);
		thespinnerepsilon.setEnabled(btraintest);
		thespinnerepsilondiff.setEnabled(btraintest);
		seedLabel.setEnabled(btraintest);
		thespinnerseed.setEnabled(btraintest);
		nodepenaltyLabel.setEnabled(!btraintest);
		thespinnernodepenalty.setEnabled(!btraintest);
	}

	/**
	 * Makes the panel showing the miRNA model options
	 */
	protected JComponent makeMIRNAPanel() {
		JPanel p = new JPanel();
		p.setBackground(lightBlue);

		BoxLayout layout = new BoxLayout(p, BoxLayout.Y_AXIS);
		p.setLayout(layout);

		JPanel input = new JPanel(new SpringLayout());

		input.setBackground(lightBlue);

		JLabel miRNASource = new JLabel("miRNA-Gene Interaction Source:");
		miRNASource.setBackground(lightBlue);
		miRNACB = new JComboBox(miRNAorganisms);
		miRNACB.setSelectedIndex(0);
		miRNACB.addActionListener(this);
		input.add(miRNASource);
		input.add(miRNACB);
		input.add(new JLabel(""));
		input.add(miRNAComboBoxHButton);
		miRNAComboBoxHButton.addActionListener(this);
		miRNAComboBoxHButton.setAlignmentX(Component.LEFT_ALIGNMENT);

		// Browsing field for miRNA interaction data
		JLabel miRNAInteractionLabel = new JLabel(
				"microRNA-Gene Interaction File:");
		miRNAInteractionField = new JTextField(miRNAInteractionDataFile,
				JLabel.TRAILING);
		miRNAInteractionField.setColumns(NUMCOLS);

		input.add(miRNAInteractionLabel);
		miRNAInteractionLabel.setLabelFor(miRNAInteractionField);
		input.add(miRNAInteractionField);
		input.add(staticmiRNAInteractionFileButton);
		input.add(staticmiRNAInterHButton);
		staticmiRNAInterHButton.addActionListener(this);

		staticmiRNAInteractionFileButton.addActionListener(this);
		// input.add(staticFileHButton);
		JLabel miRNAExpressionLabel = new JLabel(
				"microRNA Expression Data File:");
		miRNAExpressionField = new JTextField(miRNAExpressionDataFile,
				JLabel.TRAILING);
		miRNAExpressionField.setColumns(NUMCOLS);

		input.add(miRNAExpressionLabel);
		miRNAExpressionLabel.setLabelFor(miRNAExpressionField);

		input.add(miRNAExpressionField);
		input.add(staticmiRNAExpressionFileButton);
		staticmiRNAExpressionFileButton.addActionListener(this);
		input.add(staticmiRNAExpHButton);
		staticmiRNAExpHButton.addActionListener(this);

		miRNARepeatButton.setPreferredSize(new Dimension(175, 28));
		miRNARepeatButton.setMinimumSize(new Dimension(175, 28));
		miRNARepeatButton.setMaximumSize(new Dimension(175, 28));
		miRNARepeatButton.addActionListener(this);
		if (DREM_IO.miRNARepeatFilesDEF.size() >= 1) {
			miRNARepeatButton.setBackground(ListDialog.buttonColor);
		}
		input.add(new JLabel(""));
		input.add(miRNARepeatButton);
		input.add(new JLabel(""));
		input.add(miRNARepeatHButton);
		miRNARepeatHButton.addActionListener(this);
		p.add(input);
		SpringUtilities.makeCompactGrid(input, 4, 4, 4, 4, 5, 5);
		JPanel miRNAOptionsPanel = new JPanel(new SpringLayout());
		miRNAOptionsPanel.setBackground(lightBlue);
		miRNAlognormButton = new JRadioButton("Log normalize data");
		miRNAnormButton = new JRadioButton("Normalize data");
		miRNAnonormButton = new JRadioButton("No normalization/add 0");
		miRNAlognormButton.setBackground(lightBlue);
		miRNAnormButton.setBackground(lightBlue);
		miRNAnonormButton.setBackground(lightBlue);
		if (nnormalizeDEF == 0) {
			miRNAlognormButton.setSelected(true);
		} else if (nnormalizeDEF == 1) {
			miRNAnormButton.setSelected(true);
		} else {
			miRNAnonormButton.setSelected(true);
		}

		miRNAOptionsPanel.add(miRNAlognormButton);
		miRNAOptionsPanel.add(miRNAnormButton);
		miRNAOptionsPanel.add(miRNAnonormButton);
		miRNAOptionsPanel.add(miRNANormHButton);
		miRNANormHButton.addActionListener(this);

		miRNAnormGroup.add(miRNAlognormButton);
		miRNAnormGroup.add(miRNAnormButton);
		miRNAnormGroup.add(miRNAnonormButton);

		SpringUtilities.makeCompactGrid(miRNAOptionsPanel, 1, 4, 5, 5, 5, 5);
		p.add(miRNAOptionsPanel);

		JPanel filtermiRNAPanel = new JPanel(new SpringLayout());
		JLabel filtermiRNALabel = new JLabel("Filter miRNA with no"
				+ " expression from regulator data: ", JLabel.TRAILING);
		filtermiRNABox = new JCheckBox();
		filtermiRNAPanel.setBackground(lightBlue);
		filtermiRNALabel.setBackground(lightBlue);
		filtermiRNABox.setBackground(lightBlue);
		filtermiRNAPanel.add(filtermiRNALabel);
		filtermiRNAPanel.add(filtermiRNABox);
		filtermiRNAPanel.add(miRNAFilterHButton);
		miRNAFilterHButton.addActionListener(this);
		p.add(filtermiRNAPanel);
		SpringUtilities.makeCompactGrid(filtermiRNAPanel, 1, 3, 5, 5, 5, 5);


		p.add(Box.createRigidArea(new Dimension(0,50)));
		p.add(Box.createVerticalGlue());
		
		return p;
	}
	
	/**
	 * Makes the panel showing the miRNA model options
	 */
	protected JComponent makeExpressionScalingPanel() {
		JPanel p = new JPanel();
		p.setBackground(lightBlue);

		BoxLayout layout = new BoxLayout(p, BoxLayout.Y_AXIS);
		p.setLayout(layout);

		miRNACheckBox = new JCheckBox("micro RNA", checkStatusmiRNA);
		TfCheckBox = new JCheckBox("transcription factor", checkStatusTF);

		JPanel ponto = new JPanel(new SpringLayout());
		JLabel regLabel = new JLabel(
				"Incorporate expression in regulator data for: ",
				JLabel.TRAILING);
		ponto.add(regLabel);
		ponto.setBackground(lightBlue);
		TfCheckBox.setBackground(lightBlue);
		miRNACheckBox.setBackground(lightBlue);

		ponto.add(TfCheckBox);
		//TODO: ADD BACK IN THE FOLLOWING LINE WHEN MIRNA ARE IMPLEMENTED
		ponto.add(miRNACheckBox);
		ponto.add(miRNAScoringHButton);
		miRNAScoringHButton.addActionListener(this);
		p.add(ponto);
		
		//TODO: FIX THE grid when replacing above line
		SpringUtilities.makeCompactGrid(ponto, 1, 4, 5, 5, 5, 5);
		//SpringUtilities.makeCompactGrid(ponto, 1, 3, 5, 5, 5, 5);

		SpinnerNumberModel miRNAWeightModel = new SpinnerNumberModel(miRNAWeight,
				new Double(0), null, new Double(0.1));
		spinnerMIRNAWeight = new JSpinner(miRNAWeightModel);

		spinnerMIRNAWeight.setPreferredSize(new Dimension(50, 20));
		spinnerMIRNAWeight.setMinimumSize(new Dimension(50, 20));
		spinnerMIRNAWeight.setMaximumSize(new Dimension(50, 20));

		// For Tfs
		SpinnerNumberModel TFWeightModel = new SpinnerNumberModel(tfWeight,
				new Double(0), null, new Double(0.1));
		spinnerTFWeight = new JSpinner(TFWeightModel);
		spinnerTFWeight.setPreferredSize(new Dimension(50, 20));
		spinnerTFWeight.setMinimumSize(new Dimension(50, 20));
		spinnerTFWeight.setMaximumSize(new Dimension(50, 20));

		JPanel miRNAPanel = new JPanel(new SpringLayout());
		JLabel miRNALabel = new JLabel(new String(
				"Expression scaling weight"), JLabel.TRAILING);
		miRNAPanel.setBackground(lightBlue);
		miRNAPanel.add(miRNALabel);
		miRNAPanel.add(spinnerMIRNAWeight);
		miRNAPanel.add(miRNAWeightHButton);
		miRNAWeightHButton.addActionListener(this);

		JLabel TFLabel = new JLabel(new String(
				"minimum TF expression after scaling"), JLabel.TRAILING);

		miRNAPanel.add(TFLabel);
		miRNAPanel.add(spinnerTFWeight);
		miRNAPanel.add(miRNATFWeightHButton);
		miRNATFWeightHButton.addActionListener(this);
		
		p.add(miRNAPanel);
		
		SpringUtilities.makeCompactGrid(miRNAPanel, 1, 6, 5, 5, 5, 5);

		return p;
	}

	protected JComponent makeSDREMPanel() {
		JPanel p = new JPanel();
		p.setBackground(lightBlue);
		
		BoxLayout layout = new BoxLayout(p, BoxLayout.Y_AXIS);
		p.setLayout(layout);
		
		JPanel input = new JPanel(new SpringLayout());
		input.setBackground(lightBlue);

		JLabel regScoreLabel = new JLabel("Regulator Scoring File:");
		regScoreField = new JTextField(regScoreFile, JLabel.TRAILING);
		regScoreField.setColumns(NUMCOLS);

		input.add(regScoreLabel);
		regScoreLabel.setLabelFor(regScoreField);
		input.add(regScoreField);
		input.add(regScoreFileButton);
		input.add(regScoreHButton);

		regScoreHButton.addActionListener(this);
		regScoreFileButton.addActionListener(this);

		SpringUtilities.makeCompactGrid(input, 1, 4, 5, 5, 5, 5);
		p.add(input);

		SpinnerNumberModel dProbBind = new SpinnerNumberModel(new Double(0.8),
				new Double(0), null, new Double(0.1));
		spinnerdProbBind = new JSpinner(dProbBind);

		spinnerdProbBind.setPreferredSize(new Dimension(50, 20));
		spinnerdProbBind.setMinimumSize(new Dimension(50, 20));
		spinnerdProbBind.setMaximumSize(new Dimension(50, 20));
		
		JPanel inputSpin = new JPanel(new SpringLayout());
		inputSpin.setBackground(lightBlue);
		inputSpin.add(new JLabel(""));
		inputSpin.add(new JLabel("Active TF Influence: "));
		inputSpin.add(spinnerdProbBind);
		inputSpin.add(new JLabel(""));
		

		SpringUtilities.makeCompactGrid(inputSpin, 1, 4, 5, 5, 5, 5);
		p.add(inputSpin);
		p.add(Box.createRigidArea(new Dimension(0,200)));
		p.add(Box.createVerticalGlue());

		return p;
	}

	protected JComponent makeDECODPanel() {
		JPanel p = new JPanel();
		p.setBackground(lightBlue);

		JPanel input = new JPanel(new SpringLayout());
		input.setBackground(lightBlue);

		JLabel geneFastaLabel = new JLabel("Gene to Fasta Format File:");
		fastaDataField = new JTextField(fastaFile, JLabel.TRAILING);
		fastaDataField.setColumns(NUMCOLS);

		input.add(geneFastaLabel);
		geneFastaLabel.setLabelFor(fastaDataField);
		input.add(fastaDataField);
		input.add(fastaDataFileButton);
		input.add(fastaDataHButton);

		fastaDataHButton.addActionListener(this);
		fastaDataFileButton.addActionListener(this);

		JLabel decodLabel = new JLabel("Path to DECOD Executable:");
		decodPathField = new JTextField(decodPath, JLabel.TRAILING);
		decodPathField.setColumns(NUMCOLS);

		input.add(decodLabel);
		decodLabel.setLabelFor(decodPathField);
		input.add(decodPathField);
		input.add(decodPathButton);
		input.add(decodPathHButton);

		decodPathHButton.addActionListener(this);
		decodPathButton.addActionListener(this);

		SpringUtilities.makeCompactGrid(input, 2, 4, 5, 5, 5, 5);

		p.add(input);
		return p;
	}

	/**
	 * Makes the option panel on the input interface controlling the search
	 * options
	 */
	protected JComponent makeSearchPanel() {
		JPanel p = new JPanel();
		BoxLayout layout = new BoxLayout(p, BoxLayout.Y_AXIS);
		p.setLayout(layout);

		allowmergecheck = new JCheckBox(szallowmerge, ballowmergeDEF);
		JPanel pallowmergepanel = new JPanel();
		pallowmergepanel.add(allowmergecheck);
		pallowmergepanel.add(allowmergeHButton);
		allowmergecheck.setBackground(lightBlue);
		pallowmergepanel.setBackground(lightBlue);
		allowmergeHButton.addActionListener(this);
		p.add(pallowmergepanel);
		JPanel pspinnerpanel = new JPanel(new SpringLayout());
		pspinnerpanel.setBackground(lightBlue);

		SpinnerNumberModel snseed = new SpinnerNumberModel(
				new Integer(nSEEDDEF), new Integer(0), null, new Integer(1));
		thespinnerseed = new JSpinner(snseed);
		thespinnerseed.setPreferredSize(new Dimension(50, 20));
		thespinnerseed.setMinimumSize(new Dimension(50, 20));
		thespinnerseed.setMaximumSize(new Dimension(50, 20));

		SpinnerNumberModel snnumchild = new SpinnerNumberModel(new Integer(
				numchildDEF), new Integer(2), null, new Integer(1));
		thespinnernumchild = new JSpinner(snnumchild);
		thespinnernumchild.setPreferredSize(new Dimension(50, 20));
		thespinnernumchild.setMinimumSize(new Dimension(50, 20));
		thespinnernumchild.setMaximumSize(new Dimension(50, 20));

		staticsearchcheck = new JCheckBox(szstaticsearch, bstaticsearchDEF);
		JPanel pstaticsearchpanel = new JPanel();
		pstaticsearchpanel.add(staticsearchcheck);
		pstaticsearchpanel.add(staticsearchHButton);
		staticsearchcheck.setBackground(lightBlue);
		pstaticsearchpanel.setBackground(lightBlue);
		staticsearchHButton.addActionListener(this);

		pspinnerpanel.setBackground(lightBlue);

		numchildLabel = new JLabel(sznumchild, JLabel.TRAILING);
		pspinnerpanel.add(numchildLabel);
		pspinnerpanel.add(thespinnernumchild);
		pspinnerpanel.add(numchildHButton);

		numchildHButton.addActionListener(this);

		SpringUtilities.makeCompactGrid(pspinnerpanel, 1, 3, 5, 5, 5, 5);

		JPanel pinitsearchPanel = new JPanel();
		initopenButton = new JRadioButton("Use As Is");
		initopenButton.setBackground(lightBlue);
		initsearchButton = new JRadioButton("Start Search From");
		initsearchButton.setBackground(lightBlue);
		initnoButton = new JRadioButton("Do Not Use");
		initnoButton.setBackground(lightBlue);
		ButtonGroup group = new ButtonGroup();
		ninitsearchval = ninitsearchDEF;
		if (ninitsearchval == 0) {
			initopenButton.setSelected(true);
		} else if (ninitsearchval == 1) {
			initsearchButton.setSelected(true);
		} else {
			initnoButton.setSelected(true);
		}
		pinitsearchPanel.setBackground(lightBlue);
		group.add(initopenButton);
		group.add(initsearchButton);
		group.add(initnoButton);
		pinitsearchPanel.add(savedModelOptionsLabel);
		pinitsearchPanel.add(initopenButton);
		pinitsearchPanel.add(initsearchButton);
		pinitsearchPanel.add(initnoButton);
		pinitsearchPanel.add(initsearchHButton);
		initsearchHButton.addActionListener(this);
		pspinnerpanel.setBackground(lightBlue);

		p.add(pspinnerpanel);
		p.add(pstaticsearchpanel);
		p.add(pinitsearchPanel);

		SpinnerNumberModel snconvergence = new SpinnerNumberModel(new Double(
				dCONVERGENCEDEF), new Double(0.0001), null, new Double(0.001));
		thespinnerconvergence = new JSpinner(snconvergence);
		thespinnerconvergence.setPreferredSize(new Dimension(50, 20));
		thespinnerconvergence.setMinimumSize(new Dimension(50, 20));
		thespinnerconvergence.setMaximumSize(new Dimension(50, 20));

		JPanel pconvergencePanel = new JPanel(new SpringLayout());
		pconvergencePanel.setBackground(lightBlue);
		convergenceJLabel = new JLabel(szconvergence, JLabel.TRAILING);
		pconvergencePanel.add(convergenceJLabel);
		pconvergencePanel.add(thespinnerconvergence);
		pconvergencePanel.add(convergencePanelHButton);
		SpringUtilities.makeCompactGrid(pconvergencePanel, 1, 3, 5, 5, 5, 5);
		convergencePanelHButton.addActionListener(this);
		p.add(pconvergencePanel);

		SpinnerNumberModel snminstddev = new SpinnerNumberModel(new Double(
				dMINSTDDEVALDEF), new Double(0), null, new Double(0.01));
		thespinnerminstddev = new JSpinner(snminstddev);
		thespinnerminstddev.setPreferredSize(new Dimension(50, 20));
		thespinnerminstddev.setMinimumSize(new Dimension(50, 20));
		thespinnerminstddev.setMaximumSize(new Dimension(50, 20));

		JPanel minstddevPanel = new JPanel(new SpringLayout());
		minstddevPanel.setBackground(lightBlue);
		minstddevJLabel = new JLabel(szminstddev, JLabel.TRAILING);
		minstddevPanel.add(minstddevJLabel);
		minstddevPanel.add(thespinnerminstddev);
		minstddevPanel.add(minstddevPanelHButton);
		SpringUtilities.makeCompactGrid(minstddevPanel, 1, 3, 5, 5, 5, 5);
		minstddevPanelHButton.addActionListener(this);
		p.add(minstddevPanel);
		p.add(Box.createRigidArea(new Dimension(0, 35)));

		return p;
	}

	/**
	 * Makes the option panel on the input interface controlling gene filtering.
	 */
	protected JComponent makeFilterPanel() {
		staticcheck = new JCheckBox(szstaticcheck, bfilterstaticDEF);
		JPanel pstatic = new JPanel();
		staticcheck.setBackground(lightBlue);
		pstatic.setBackground(lightBlue);
		pstatic.add(staticcheck);
		pstatic.add(staticHButton);
		staticHButton.addActionListener(this);

		JPanel p = new JPanel(new SpringLayout());

		p.add(maxmissingLabel);
		p.add(thespinnermaxmissing);
		p.add(maxmissingHButton);
		maxmissingHButton.addActionListener(this);
		p.add(filterthresholdLabel);
		p.add(thespinnerfilterthreshold);
		p.add(filterthresholdHButton);
		filterthresholdHButton.addActionListener(this);
		p.add(expressLabel);
		p.add(thespinnerexpress);
		p.add(expressHButton);
		expressHButton.addActionListener(this);

		JPanel p15 = new JPanel();
		filterchoiceLabel = new JLabel(szfilterchoice, JLabel.TRAILING);
		p15.add(filterchoiceLabel);
		maxminButton = new JRadioButton("Maximum\u2212Minimum");
		maxminButton.setBackground(lightBlue);
		absButton = new JRadioButton("Difference from 0");
		absButton.setBackground(lightBlue);

		ButtonGroup group = new ButtonGroup();
		if (bmaxminval) {
			maxminButton.setSelected(true);
		} else {
			absButton.setSelected(true);
		}
		p15.setBackground(lightBlue);
		group.add(maxminButton);
		group.add(absButton);
		JPanel buttonPanel = new JPanel();
		buttonPanel.add(maxminButton);
		buttonPanel.add(absButton);
		buttonPanel.setBackground(lightBlue);
		p15.add(buttonPanel);
		p15.add(filterchoiceHButton);
		filterchoiceHButton.addActionListener(this);

		SpringUtilities.makeCompactGrid(p, 3, 3, 5, 5, 5, 5);

		JPanel p2 = new JPanel();
		p2.add(extraLabel);
		p2.add(extraField);

		extraField.setPreferredSize(new Dimension((int) orig1Field
				.getPreferredSize().getWidth(), (int) extraButton
				.getPreferredSize().getHeight()));

		p2.add(extraButton);
		p2.add(extraHButton);
		p2.setBackground(lightBlue);
		extraButton.addActionListener(this);
		extraHButton.addActionListener(this);

		JPanel entirepanel = new JPanel();

		BoxLayout layout = new BoxLayout(entirepanel, BoxLayout.Y_AXIS);

		entirepanel.setLayout(layout);
		p.setBackground(lightBlue);
		entirepanel.add(pstatic);
		entirepanel.add(p);
		entirepanel.add(p15);
		entirepanel.add(p2);

		entirepanel.setBackground(lightBlue);

		return entirepanel;
	}

	/**
	 * Create the GUI and show it. For thread safety, this method should be
	 * invoked from the event-dispatching thread.
	 */
	private static void createAndShowGUI() throws FileNotFoundException,
			IOException {
		// Make sure we have nice window decorations.
		// JFrame.setDefaultLookAndFeelDecorated(true);

		// Create and set up the window.

		JFrame frame = new DREM_IO();
		frame.setLocation(10, 25);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

		// Display the window.
		frame.pack();
		frame.setVisible(true);
	}

	//TODO: remove when io_batch is complete
	/**
	 * Run DREM in batch mode programmatically.
	 * 
	 * @param defaults
	 *            the file specifiying all parameter values.
	 * @param bindFile
	 *            the TF-gene binding data (optionaly with priors)
	 * @param outputFile
	 *            the filename that will be used when saving the model and TF
	 *            activity scores
	 * @deprecated Replaced by DREM_IO_Batch
	 */
	public static void batchMode(String defaults, String bindFile,
			String outputFile) throws IOException, FileNotFoundException {
		// The active TF inluence parameter is specified in the DREM defaults
		// file
		szDefaultFile = defaults;
		DREM_IO drem = new DREM_IO();

		// Make sure to set after calling the constructor, which
		// initializes it to false
		drem.bbatchMode = true;
		drem.saveFile = outputFile;
		drem.staticFileField.setText(bindFile);
		drem.clusterAButton.doClick();
	}

	/**
	 * The main method which when executed will have the input interface created
	 */
	public static void main(String[] args) throws Exception {

		boolean bshowusage = true;
		boolean bbatchmode = false;
		String szoutmodelfile = "";

		if (args.length == 0) {
			bshowusage = false;
		} else if (args.length == 2) {
			if (args[0].equals("-d")) {
				szDefaultFile = args[1];
				bshowusage = false;
			}
		} else if (args.length == 3) {
			if (args[0].equals("-b")) {
				szDefaultFile = args[1];
				szoutmodelfile = args[2];
				bshowusage = false;
				bbatchmode = true;
			}
		}

		if (bshowusage) {
			System.out
					.println("USAGE: java -jar drem.jar [-d defaultfilename.txt|-b settingsfile.txt outmodelfile.txt]");
			return;
		}

		// Schedule a job for the event-dispatching thread:
		// creating and showing this application's GUI.
		if (bbatchmode) {
			new DREM_IO_Batch(szDefaultFile, szoutmodelfile);// szBatchInputDir,szBatchOutputDir);
		} else {
			javax.swing.SwingUtilities.invokeLater(new Runnable() {
				public void run() {
					try {
						createAndShowGUI();
					} catch (FileNotFoundException ex) {
						ex.printStackTrace(System.out);
					} catch (IOException ex) {
						ex.printStackTrace(System.out);
					}
				}
			});
		}
	}
        
        //FIXME: integrate methylation to TF binding to update gene2RegBinding, gene2RegBindingIndex;
        public HashMap<String,ArrayList<String>> methy2Bind(DREM_DataSet theDataSet, String szExFile, DataSetCore miRNADataSet, 
                String djMethyFile,String djMethyGTF){
            
            
            HashMap<String,ArrayList<String>> peakGeneMap=new HashMap<>();
            HashMap<String,ArrayList<String>> silPeakGeneMap=new HashMap<>();
            // read in methylation peaks 
            try{                
                BufferedReader br=new BufferedReader(new FileReader(djMethyFile));
                String Line;
                String[] LineSplit;
                double dgDistance;
                double DISTANCECUT=10000;
                String dg;
             
                Line=br.readLine();
                String timeName;
                
                //read in peak file 
                while ((Line=br.readLine())!=null){
                    Line=Line.trim();
                    LineSplit=Line.split("\\\t");
                    timeName=LineSplit[3].split("_")[0];
                    dgDistance=Double.parseDouble(LineSplit[LineSplit.length-1]);
                    dg=LineSplit[LineSplit.length-2].toUpperCase();
                    if ((dgDistance<DISTANCECUT) &&(dg.equals("")==false)){
                        if (peakGeneMap.containsKey(timeName)==false) {
                            ArrayList<String> PeakGeneList=new ArrayList<>();
                            PeakGeneList.add(dg);
                            peakGeneMap.put(timeName,PeakGeneList);
                        }else{
                            if (!peakGeneMap.get(timeName).contains(dg)){
                                peakGeneMap.get(timeName).add(dg);
                            }
                        }         
                    } 
                }
                //determine whether it's sienced based on expression 
                silPeakGeneMap=getSilencedGene(peakGeneMap,theDataSet,szExFile);
                //write silPeakGeneMap to file
                //writeHashMap(silPeakGeneMap,"silGeneMap.txt");
                
            }catch (IOException e){
                return silPeakGeneMap;
                //e.printStackTrace();  
            }
            return silPeakGeneMap;
        }
        
         //get methy promoter scores
        public HashMap<String,HashMap<String,Double>> getMethyScore(String djMethyFile, String djMethyGTF){
            HashMap<String,HashMap<String,Double>> methyScoreMap=new HashMap<>(); //methy Score for gene promoter 
            try{   
                
                BufferedReader br=new BufferedReader(new FileReader(djMethyFile));
                String Line;
                String[] LineSplit;
                double mScore;
                String dg;
                String key;
             
                Line=br.readLine();
                String timeName;
                
                //read in peak file get closet peak for each gene
                while ((Line=br.readLine())!=null){
                    Line=Line.trim();
                    LineSplit=Line.split("\\\t");
                    timeName=LineSplit[3].split("_")[0];
                    dg=LineSplit[3].split("_")[1].toUpperCase();
                    mScore=Double.parseDouble(LineSplit[4]);
                    //key=timeName+','+dg.toUpperCase();
                    if (!methyScoreMap.containsKey(timeName)){
                        HashMap<String,Double> tmpTimeMap=new HashMap<>();
                        tmpTimeMap.put(dg, mScore);
                        methyScoreMap.put(timeName,tmpTimeMap);
                    }else{
                        methyScoreMap.get(timeName).put(dg,mScore);
                    }
                  
                }    
             
            }catch (IOException e){
                e.printStackTrace();  
            }
            return methyScoreMap;
        }
        
        
         //getSilencedGene
        public HashMap<String, ArrayList<String>> getSilencedGene(HashMap<String, ArrayList<String>> peakGeneMap,
                DREM_DataSet theDataSet, String szExFile ) {
            // it was based on two strategy:
            //(1) absolute expression (very low, bottom 10%) --> to handle those already silenced genes at the begining. 
            //(2) large down-regualation --> capture silenced genes during the process 
            //
            HashMap<String,ArrayList<String>> peakSilGeneMap= new HashMap<>();
            String[] GeneTimePoints=getTimePoints(szExFile,theDataSet.badd0);
            // read absolute expression from expression file
            Map<String,Double> AbEx=new HashMap<>();
            AbEx=getAbEx(szExFile);
            java.util.List<Double> AbExVal=new ArrayList<Double>(AbEx.values());
            
            for (String keyTime: peakGeneMap.keySet()){
                ArrayList<String> valueGene=peakGeneMap.get(keyTime);
                double[] vgene_ex;
                double vgene_ex_rel;  //relative gene expression ==> down-regualted genes
                double vgene_ex_rel_cut=-1; // realative gene expression cutoff 
                        
                double vgene_ex_ab;   //aboslute gene expression ==> low expressed genes 
                double vgene_ex_ab_index;
                double vgene_ex_ab_per;
                double vgene_ex_ab_cut=0.4; // aboslute gene expression cutoff 
                
                int keyTimeIndex=Arrays.asList(GeneTimePoints).indexOf(keyTime);
                if(keyTimeIndex!=-1){
                    for (String vgene:valueGene){
                        vgene=vgene.toUpperCase();
                        String vkey=keyTime+','+vgene;
                        int vGeneIndex=Arrays.asList(theDataSet.genenames).indexOf(vgene);
                        if (vGeneIndex!=-1){
                            vgene_ex=theDataSet.data[vGeneIndex];
                            if (keyTimeIndex!=0){
                                vgene_ex_rel=vgene_ex[keyTimeIndex]-vgene_ex[keyTimeIndex-1];
                                vgene_ex_ab=AbEx.get(vkey);
                                vgene_ex_ab_index=AbExVal.indexOf(vgene_ex_ab);
                                vgene_ex_ab_per=vgene_ex_ab_index/AbEx.size();
                                if ((vgene_ex_ab_per>vgene_ex_ab_cut)|| (vgene_ex_rel<vgene_ex_rel_cut)){
                                    if (!peakSilGeneMap.containsKey(keyTime)){
                                        ArrayList<String> PeakGeneList=new ArrayList<>();
                                        PeakGeneList.add(vgene);
                                        peakSilGeneMap.put(keyTime,PeakGeneList);
                                    }else{
                                        if (!peakSilGeneMap.get(keyTime).contains(vgene)){
                                           peakSilGeneMap.get(keyTime).add(vgene);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            return peakSilGeneMap;
        }
        
         //FIXME：read gene expression from expression file
        public Map<String,Double> getAbEx(String FileName){
            HashMap<String,Double> abEx=new HashMap<>();
            Map<String,Double> sortAbEx=new HashMap<>();
            try{
                BufferedReader br=new BufferedReader(new FileReader(FileName));
                String Line;
                String[] FR=br.readLine().split("\\\t");
                String[] LineSplit;
                while((Line=br.readLine())!=null){
                    LineSplit=Line.split("\\\t");
                    String LineGene=LineSplit[0];
                    for (int i=1;i<FR.length;i++){
                        String itime=FR[i];
                        String key=itime+','+LineGene.toUpperCase();
                        double val=Double.parseDouble(LineSplit[i]);
                        abEx.put(key,val);
                    }
                }
                sortAbEx=sortByValue(abEx);
                //System.out.print("test");
            }catch (IOException e){
                e.printStackTrace();
            }
            return sortAbEx;
        }
         
        
         public String[] getTimePoints(String FileName, Boolean badd0){
            String[] TimePoints=null;
            ArrayList<String> FTimePoints=new ArrayList<>();
            try{
                    BufferedReader br=new BufferedReader(new FileReader(FileName));
                    String FR=br.readLine();
                    TimePoints=FR.split("\\\t");
                    TimePoints=Arrays.copyOfRange(TimePoints,1,TimePoints.length);
                    
                    if (badd0){
                        FTimePoints.add("Added_0");
                    }
                    for (String elem : TimePoints){
                        FTimePoints.add(elem);
                    }
                   TimePoints=FTimePoints.toArray(new String[FTimePoints.size()]);
            }catch (IOException e){
                e.printStackTrace();
            }
            return TimePoints;
        }
        
        /*
          public String[] getTimePoints(String FileName){
            String[] TimePoints=null;
            try{
                    BufferedReader br=new BufferedReader(new FileReader(FileName));
                    String FR=br.readLine();
                    TimePoints=FR.split("\\\t");
                    TimePoints=Arrays.copyOfRange(TimePoints,1,TimePoints.length);
            }catch (IOException e){
                e.printStackTrace();
            }
            return TimePoints;
        }
          
        **/
        
        // sort HashMap based on the values
        private static Map<String, Double> sortByValue(Map<String, Double> unsortMap) {

            // 1. Convert Map to List of Map
            java.util.List<Map.Entry<String, Double>> list;
            list = new LinkedList<Map.Entry<String, Double>>(unsortMap.entrySet());

            // 2. Sort list with Collections.sort(), provide a custom Comparator
            //    Try switch the o1 o2 position for a different order
            Collections.sort(list, new Comparator<Map.Entry<String, Double>>() {
                public int compare(Map.Entry<String, Double> o1,
                                   Map.Entry<String, Double> o2) {
                    return (o1.getValue()).compareTo(o2.getValue());
                }
            });
                    
            Map<String, Double> sortedMap = new LinkedHashMap<String, Double>();
            for (Map.Entry<String, Double> entry : list) {
                sortedMap.put(entry.getKey(), entry.getValue());
            }
            return sortedMap;
        }
        
        //write hashmap
         public void writeHashMap(HashMap<String,ArrayList<String>> silPeakGeneMap,String outname){
            StringBuffer outstrbuffer=new StringBuffer();
            for (String key:silPeakGeneMap.keySet()){
                ArrayList<String> Val=new ArrayList<>();
                Val=silPeakGeneMap.get(key);
                String ValStr="";
                for (String ValItem: Val){
                    ValStr=ValStr+ValItem+",";
                }
                outstrbuffer.append(key+"\t"+ValStr+"\n");
            }
            
            String outstr=outstrbuffer.toString();
            try{
                BufferedWriter br=new BufferedWriter(new FileWriter(outname));
                br.write(outstr);
                br.close();
            }catch (IOException e){
                 e.printStackTrace();  
            }
        }
        
        
}

