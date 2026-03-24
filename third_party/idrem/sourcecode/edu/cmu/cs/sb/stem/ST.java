package edu.cmu.cs.sb.stem;

import edu.cmu.cs.sb.core.*;
import edu.cmu.cs.sb.chromviewer.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.filechooser.*;
import javax.swing.*;
import javax.swing.event.*;
import javax.swing.text.*;
import java.util.*;
import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeEvent;
import java.awt.image.*;
import java.text.*;
import java.net.*;


/**
 *Class implementing the main input interface
 */
public class ST extends JFrame implements ActionListener
{
    boolean boutputdir;
    String szBatchInputDir;
    String szBatchOutputDir;
    String szcurrentDefaultFile;
    JDialog executeDialognf= null;

    String szorganismsourceval;
    String szxrefsourceval;
    String szchromsourceval;
   
    boolean[] bdownloading = new boolean[4];
    Object lockpd = new Object();
    int[] npercentdone = new int[4];
    boolean[] bexception = new boolean[4];
    int nexceptions;
    static Object xreflock = new Object();
    JPanel pcluster;	
	   
    static String EBIURL = "ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/";
    static String szCompareDEF = "";
    static Vector vRepeatCompareDEF = new Vector();
    static int nCompareMinGenesDEF = 5;
    static double dCompareMinpvalDEF = 0.005;
    static boolean bcomparealltimeDEF = true;

    static String sztaxonDEF = "";
    static String szevidenceDEF = "";
    static String szDefaultFile = "";
    static String szBatchGOoutput =null;

    static String szDataFileDEF = "";
    static String szCrossRefFileDEF = "";
    static String szGeneAnnotationFileDEF = "";
    static String szChromFileDEF = "";
    static String  szGeneOntologyFileDEF = "gene_ontology.obo";
    static int   nMaxProfilesDEF = 50;
    static int ngenedisplayDEF = 0;
    static int ngenescaleDEF = 1;
    static int ngenescalekmeansDEF = 2;
    static double dyscaleminDEF = -3;
    static double dyscalemaxDEF = 3;
    static double dtickDEF = 1;
    static boolean bautomaticYDEF = true;
    static boolean buniformXaxisDEF = true;
    static boolean bscalevisibleDEF = true;
    static boolean bdisplayIDDEF = true; 
    static boolean bdisplayprofilelineDEF = true; 
    static boolean bdisplaymaintickDEF = false; 
    static boolean bdisplaydetailDEF = true; 
    static double dtickmainDEF = 1;

    static int nKDEF = 10;
    static int nREPDEF = 20;
    static int nCLUSTERINGMETHODDEF = 0;
    static int   nMaxUnitDEF = 2;
    static Vector  vRepeatFilesDEF = new Vector();

    static boolean  balltimeDEF = true;
    static int   nMaxMissingDEF = 0;
    static int genecolorRDEF = 204;
    static int genecolorGDEF = 51;
    static int genecolorBDEF = 0;
    static double dMinExpressionDEF = 1;
    static double dMinCorrelationRepeatsDEF = 0;
    static String  szPrefilteredDEF = "";
    static double  dMaxCorrelationModelDEF = 1;
    static int nNumPermsGeneDEF = 50;
    static int nMaxCandidateModelDEF = 1000000;
    static double dSignificanceLevelDEF = .05;
    static int nfdrDEF = 2;
    static double dMinimumCorrelationClusteringDEF = .7; 
    static double  dMinimumPercentileClusteringDEF = 0; 
    static int nSamplesMultipleDEF = 500;
    static int nMinGoGenesDEF = 5;
    static int nMinGOLevelDEF = 3; 
    static int ndbDEF = 0;
    static int nxrefDEF = 0;
    static int nnormalizeDEF = 1;
    static int nlocationDEF = 0;

    static boolean bspotcheckDEF = false;
    static String szxrefDEF = "";
    static String szcategoryIDDEF = "";
    static String szlocationDEF = "";
    static boolean bpontoDEF = true;
    static boolean  bcontoDEF = true;
    static boolean bfontoDEF = true;
    static boolean brandomgoDEF = true;
    static boolean bmaxminDEF = true;
    static boolean ballpermuteDEF =true;

    long s1;
    JRadioButton bfButton, fdrButton, noneButton;
    JRadioButton bfgoButton, randomgoButton;
    JRadioButton maxminButton, absButton;

    String szorig1val;
    String szchromval = szChromFileDEF;
    String szxrefval = szCrossRefFileDEF;

    String szorig2val;
    String szgoval = szGeneAnnotationFileDEF;
    String szgocategoryval=szGeneOntologyFileDEF;
    String szextraval;
    String szcategoryIDval;

    String szevidenceval;
    String sztaxonval;
    boolean bpontoval;
    boolean bcontoval;
    boolean bfontoval;
    String szmaxmissingval;
    String szexpressval;
    String szfilterthresholdval;
    String szlbval;
    String szalphaval;  
    String szpercentileval;
    String szmaxchangeval;
    String sznumberprofilesval;
    String szcorrmodelval;
    String szsamplegeneval;
    String szsamplepvalval; 
    String szsamplemodelval;
    String szmingoval;
    String szmingolevelval;
    boolean brandomgoval = true;
    boolean btakelog = false;
    boolean bmaxminval = true;
    boolean ballpermuteval =true;
    int ndb;
    int nK = nKDEF;
    int nreps  = nREPDEF;
    int nMaxProfiles = nMaxProfilesDEF;
    int nMaxChangeProfiles = nMaxUnitDEF;
    int nxrefcb;
    int nchromcb;
    int nclusteringmethodcb = nCLUSTERINGMETHODDEF;
    int noldcluteringmethodcb = -1;
    String szusergann;
    String szuserxref;
    String szuserchrom;

    int ntodownload;
    int nfdr = 0;  //0 none, 1 false discovery, 2 bonferroni

    boolean balltime = false;
    boolean bspotincluded;
    boolean badd0 = false;

    /**
     *True we run in batch mode without a GUI, false we use the GUI
     */
    boolean bbatchmode;
        
    JLabel orig1Label;
    JLabel xrefLabel;
    JLabel xrefsourceLabel;
    JLabel chromLabel;
  
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
    JLabel numberprofilesLabel;
    JLabel percentileLabel;
    JLabel lbLabel;
    JLabel samplegeneLabel;
    JLabel samplemodelLabel;
    JLabel samplepvalLabel;
    JLabel goLabel;
    JLabel permuteLabel;

    JLabel mingoLabel;
    JLabel mingolevelLabel;
    JLabel randomgoLabel;
    JLabel filterchoiceLabel;
    JComboBox chromcb = new JComboBox(chromorganisms);
    JComboBox orgcb;
    JComboBox xrefcb;
    JComboBox clusteringmethodcb;

    JRadioButton lognormButton, normButton, nonormButton;
    ButtonGroup normGroup = new ButtonGroup();

    JCheckBox chromcheck = new JCheckBox("Locations", false);
    JCheckBox anncheck = new JCheckBox("Annotations", false);
    JCheckBox xrefcheck = new JCheckBox("Cross References", false);

    JCheckBox obocheck = new JCheckBox("Ontology", false);
    JCheckBox permutecheck = new JCheckBox("Permutation Test Should Permute Time Point 0", true);

    JCheckBox pcheck; 
    JCheckBox fcheck; 
    JCheckBox ccheck;
    JCheckBox spotcheck;
      
    static int NUMCOLS = 42;


    static Color lightBlue = new Color(178,223,238);
    static Color buttonColor = new Color(255,246,143);
    static Color defaultColor;

    static String[] clusteringmethod = {"STEM Clustering Method",
					"K-means"};

    static String SZBIOMART = " (Ensembl/Biomart)";
    static String[] chromorganisms = {
                              "User provided",
			      "No Gene Locations",
			      /*
			      "Aedes aegypti"+SZBIOMART,
			      "Anopheles gambiae"+SZBIOMART,
			      "Bos taurus"+SZBIOMART,
			      "Caenorhabditis elegans"+SZBIOMART,
			      "Canis familiaris"+SZBIOMART,
			      "Cavia porcellus"+SZBIOMART,
			      "Ciona intestinalis"+SZBIOMART,
			      "Ciona savignyi"+SZBIOMART,
			      "Danio rerio"+SZBIOMART,
			      "Dasypus novemcinctus"+SZBIOMART,
			      "Drosophila melanogaster"+SZBIOMART,
			      "Echinops telfairi"+SZBIOMART,
			      "Erinaceus europaeus"+SZBIOMART,
			      "Felis catus"+SZBIOMART,
			      "Gallus gallus"+SZBIOMART,
			      "Gasterosteus aculeatus"+SZBIOMART,
			      "Homo sapiens"+SZBIOMART,
			      "Loxodonta africana"+SZBIOMART,
			      "Macaca mulatta"+SZBIOMART,
			      "Microcebus murinus"+SZBIOMART,
			      "Monodelphis domestica"+SZBIOMART,
			      "Mus musculus"+SZBIOMART,
			      "Myotis lucifugus"+SZBIOMART,
			      "Ochotona princeps"+SZBIOMART,
			      "Ornithorhynchus anatinus"+SZBIOMART,
			      "Oryctolagus cuniculus"+SZBIOMART,
			      "Oryzias latipes"+SZBIOMART,
			      "Otolemur garnettii"+SZBIOMART,
			      "Pan troglodytes"+SZBIOMART,
			      "Rattus norvegicus"+SZBIOMART,
			      "Saccharomyces cerevisiae"+SZBIOMART,
			      "Sorex araneus"+SZBIOMART,
			      "Spermophilus tridecemlineatus"+SZBIOMART,
			      "Takifugu rubripes"+SZBIOMART,
			      "Tetraodon nigroviridis"+SZBIOMART,
			      "Tupaia belangeri"+SZBIOMART,
			      "Xenopus tropicalis"+SZBIOMART
			      */
                            };



    //     static String SZGP ="gene_association.";


     static  String[] chromfile = {
                              "",
			      "",
			      /*
			      "aaegypti.mart.gz",
			      "agambiae.mart.gz",
			      "btaurus.mart.gz",
			      "celegans.mart.gz",
			      "cfamiliaris.mart.gz",
			      "cporcellus.mart.gz",
			      "cintestinalis.mart.gz",
			      "csavignyi.mart.gz",
			      "drerio.mart.gz",
			      "dnovemcinctus.mart.gz",
			      "dmelanogaster.mart.gz",
			      "etelfairi.mart.gz",
			      "eeuropaeus.mart.gz",
			      "fcatus.mart.gz",
			      "ggallus.mart.gz",
			      "gaculeatus.mart.gz",
			      "hsapiens.mart.gz",
			      "lafricana.mart.gz",
			      "mmulatta.mart.gz",
			      "mmurinus.mart.gz",
			      "mdomestica.mart.gz",
			      "mmusculus.mart.gz",
			      "mlucifugus.mart.gz",
			      "oprinceps.mart.gz.",
			      "oanatinus.mart.gz",
			      "ocuniculus.mart.gz",
			      "olatipes.mart.gz",
			      "ogarnettii.mart.gz",
			      "ptroglodytes.mart.gz",
			      "rnorvegicus.mart.gz",
			      "scerevisiae.mart.gz",
			      "saraneus.mart.gz",
			      "stridecemlineatus.mart.gz",
			      "trubripes.mart.gz",
			      "tnigroviridis.mart.gz",
			      "tbelangeri.mart.gz",
			      "xtropicalis.mart.gz"
			      */
                              };


    /*
    //currently not using 
    //for a GO organism maps to corresponding biomart location data
     static  String[] organismschrom = {
                              "",
			      "",
			      "",
			      "",
			      "",
			      "",
			      "",
			      "",
			      "Caenorhabditis elegans"+SZBIOMART,
			      "",
			      "",
			      "",
			      "",
			      "",//new
			      "",
			      //"",
			      "Gallus gallus"+SZBIOMART,
			      "",
			      "Danio rerio"+SZBIOMART,
			      "",
			      "",
			      "Drosophila melanogaster"+SZBIOMART,
			      "",//new
			      "",
			      "",
			      "",
			      "Homo sapiens"+SZBIOMART,
			      "",
			      "",
			      "",
			      "",
			      "",		      
			      //"",
			      "Mus musculus"+SZBIOMART,
			      "Mus musculus"+SZBIOMART,
			      "",//new
			      "",
			      "",
			      "",
			      "",
			      "",
			      "",
			      "",
			      "",
			      "Rattus norvegicus"+SZBIOMART,
			      "Rattus norvegicus"+SZBIOMART,
			      "",
			      "Saccharomyces cerevisiae"+SZBIOMART,
			      "",
			      "",
			      "",
			      "",
			      "",
			      "",
			      "",
			      "",
			      "",
			      "Danio rerio"+SZBIOMART};
    */

    static String szorig1 = "Data File: ";
    static String szgo = "Gene Annotation File: ";
    static String szxreflt = "Cross Reference File: ";
    static String szchromsource = "Gene Location Source: ";
    static String szxrefsource = "Cross Reference Source: ";
    static String szmaxmissing = "Maximum Number of Missing Values: ";
    static String szextra = "Pre-filtered Gene File: ";
    static String szexpress = "Minimum Absolute Expression Change: ";
    static String szfilterthreshold = "Minimum Correlation between Repeats: ";
    static String szlb = "Minimum Correlation: ";
    static String szpercentile = "Minimum Correlation Percentile (repeat only):";
    static String szalpha = "Significance Level: ";// for Model Profiles: ";
    static String szmaxchange = "Maximum Unit Change in Model Profiles between Time Points: ";
    static String szrep       = "Number of Random Starts: ";
    static String sznumberprofiles = "Maximum Number of Model Profiles: ";// (0 for no limit): ";
    static String szk = "Number of Clusters (K): ";// (0 for no limit): ";
    static String szcorrmodel = "Maximum Correlation: ";// Between Model Profiles: ";
    static String szsamplegene = "Number of Permutations per Gene (0 for all permutations): ";
    static String szsamplemodel = "Maximum Number of Candidate Model Profiles: ";
    static String szsamplepval = "Number of samples for randomized multiple hypothesis correction: ";
    static String szmingo = "Minimum number of genes: ";
    static String szmingolevel = "Minimum GO level: ";
    static String szrandomgo = "Multiple hypothesis correction method for actual size based enrichment: ";
    static String szfilterchoice = "Change should be based on:";
    static String szlogcheck = "Log normalize the input data";
    static String szcategoryID = "Category ID mapping file: ";
    static String szspotcheck = "Spot IDs included in the data file";
    static String szpermute = "Permutations should use:";
    static String szchrom = "Gene Location File: ";

    //Fields for data entry
    JTextField orig1Field;
    JTextField goField;
    JTextField chromField;
    JTextField extraField;
    JTextField xrefField;
    JTextField categoryIDField;
    JTextField taxonField;
    JTextField evidenceField;

    JButton orig1Button = new JButton("Browse...", Util.createImageIcon("Open16.gif"));
    JButton orig2Button = new JButton("Browse...", Util.createImageIcon("Open16.gif"));
    JButton chromButton = new JButton("Browse...", Util.createImageIcon("Open16.gif"));
    JButton goLabelButton = new JButton("Browse...", Util.createImageIcon("Open16.gif"));
    JButton xrefButton = new JButton("Browse...", Util.createImageIcon("Open16.gif"));
    JButton extraButton = new JButton("Browse...", Util.createImageIcon("Open16.gif"));
    JButton categoryIDButton = new JButton("Browse...", Util.createImageIcon("Open16.gif"));
    JButton clusterAButton; 
    JButton logHButton =  new JButton(Util.createImageIcon("Help16.gif"));
    JButton viewButton = new JButton("View Data File");
    JButton repeatButton = new JButton("Repeat Data...", Util.createImageIcon("Open16.gif"));
    JButton orig1HButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton orig2HButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton advancedHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton goLabelHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton categoryIDHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton extraHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton numberProfileHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton changeHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton maxmissingHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton filterthresholdHButton = new JButton(Util.createImageIcon("Help16.gif"));
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
    JButton xrefsourceHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton downloadlistgoHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton xrefHButton =  new JButton(Util.createImageIcon("Help16.gif"));
    JButton presetsHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton executeHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton spotHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton openButton = new JButton("Load Saved Settings...",Util.createImageIcon("Open16.gif"));
    JButton infoButton = new JButton(Util.createImageIcon("About16.gif"));
    JButton viewHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton evidenceHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton ontoHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton taxonHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton randomgoHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton clusteringmethodHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton optionsButton = new JButton("Advanced Options...", Util.createImageIcon("Preferences16.gif"));
    JButton filterchoiceHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton permuteHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton chromHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton chromcbHButton = new JButton(Util.createImageIcon("Help16.gif"));

    static JFileChooser fc = new JFileChooser();

    JSpinner thespinnermingo;
    JSpinner thespinnermingolevel;
    JSpinner thespinnerProfile;
    JSpinner thespinnerChange;
    JSpinner thespinnermaxmissing; 
    JSpinner thespinnerfilterthreshold;
    JSpinner thespinnerexpress;
    JSpinner thespinnercorrmodel;
    JSpinner thespinneralpha;
    JSpinner thespinnerlb;
    JSpinner thespinnerpercentile;
    JSpinner thespinnersamplegene;
    JSpinner thespinnersamplemodel;
    JSpinner thespinnersamplepval;

    JCheckBox thelogcheck;
    JDialog theOptions;
    ListDialog theRepeatList;
    String szClusterA = "Execute";

    /**
     * Constructor for running STEM in batch mode
     */
    public ST(String szBatchInputDir,String szBatchOutputDir)
    {
	Util.theChooser = new JFileChooser();
	this.bbatchmode = true;
	this.szBatchInputDir = szBatchInputDir;
	this.szBatchOutputDir = szBatchOutputDir;

	runBatchDir();
    }

    /**
     * Constructor for running STEM through the normal input interface
     * Renders the interace
     */
    public ST() throws FileNotFoundException, IOException
    {
        super("STEM: Short Time-series Expression Miner");
	Util.theChooser = new JFileChooser();
	this.bbatchmode = false;

        try
	{
           if (!szDefaultFile.trim().equals(""))
	   {
	      parseDefaults(szDefaultFile);
	   }
	}
        catch(IllegalArgumentException iex)
        {
           final IllegalArgumentException fiex = iex;
	   final JFrame fframe = this; 
           javax.swing.SwingUtilities.invokeLater(new Runnable() 
           {
              public void run() 
              {
                 JOptionPane.showMessageDialog(fframe, fiex.getMessage(), 
                                "Exception thrown", JOptionPane.ERROR_MESSAGE);
	      }
	   });
        }
        catch(FileNotFoundException iex)
        {
           final FileNotFoundException fiex = iex;
	   final JFrame fframe = this; 
           javax.swing.SwingUtilities.invokeLater(new Runnable() 
           {
              public void run() 
              {
                 JOptionPane.showMessageDialog(fframe, fiex.getMessage(), 
                                "Exception thrown", JOptionPane.ERROR_MESSAGE);
	      }
	   });
        }
        catch (Exception ex)
        {
	       final Exception fex = ex;
               final JFrame fframe = this;
              javax.swing.SwingUtilities.invokeLater(new Runnable() 
              {
                 public void run() 
                 {
                     JOptionPane.showMessageDialog(fframe, fex.toString(), 
                                "Exception thrown", JOptionPane.ERROR_MESSAGE);
                     fex.printStackTrace(System.out);
		 }
	     });
        }     

        bmaxminval = bmaxminDEF;
        ballpermuteval = ballpermuteDEF;

        orig1Label = new JLabel(szorig1, JLabel.TRAILING);
        extraLabel = new JLabel(szextra, JLabel.TRAILING);
	xrefLabel = new JLabel(szxreflt, JLabel.TRAILING);
	chromLabel = new JLabel(szchrom, JLabel.TRAILING);
        alphaLabel = new JLabel(szalpha, JLabel.TRAILING);
        lbLabel = new JLabel(szlb, JLabel.TRAILING);
        maxmissingLabel = new JLabel(szmaxmissing, JLabel.TRAILING);
        filterthresholdLabel = new JLabel(szfilterthreshold, JLabel.TRAILING);       
        expressLabel = new JLabel(szexpress, JLabel.TRAILING);

        maxchangeLabel = new JLabel(szmaxchange, JLabel.TRAILING);
        numberprofilesLabel = new JLabel(sznumberprofiles, JLabel.TRAILING);

        corrmodelLabel = new JLabel(szcorrmodel, JLabel.TRAILING);
        percentileLabel = new JLabel(szpercentile, JLabel.TRAILING);
        samplegeneLabel = new JLabel(szsamplegene, JLabel.TRAILING);
        samplemodelLabel = new JLabel(szsamplemodel, JLabel.TRAILING);
        samplepvalLabel = new JLabel(szsamplepval, JLabel.TRAILING);
        goLabel = new JLabel(szgo, JLabel.TRAILING);
	categoryIDLabel = new JLabel(szcategoryID, JLabel.TRAILING);
        categoryIDField = new JTextField(szcategoryIDDEF, JLabel.TRAILING);
        categoryIDField.setColumns(NUMCOLS-2);

        orig1Field = new JTextField(szDataFileDEF, JLabel.TRAILING);
        orig1Field.setColumns(NUMCOLS);

        xrefField = new JTextField(szCrossRefFileDEF, JLabel.TRAILING);
        xrefField.setColumns(NUMCOLS);

        chromField = new JTextField(szChromFileDEF, JLabel.TRAILING);
        chromField.setColumns(NUMCOLS);

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
        
        BoxLayout layout = new BoxLayout(contentPane,BoxLayout.Y_AXIS);
        contentPane.setLayout(layout);
        contentPane.setBackground(lightBlue);

        JPanel p = new JPanel(new SpringLayout());
	JLabel step1 = new JLabel("  1.  Expression Data Info:");
	p.add(step1);
	p.add(new JLabel(""));
	p.add(new JLabel(""));
	p.add(new JLabel(""));
        p.setBackground(lightBlue);
        p.add(orig1Label);
        p.add(orig1Field);
        p.add(orig1Button);
        p.add(orig1HButton);

        orig1Button.addActionListener(this);
        orig1HButton.addActionListener(this);

        repeatButton.setPreferredSize(new Dimension(175,26));
        repeatButton.setMinimumSize(new Dimension(175,26));
        repeatButton.setMaximumSize(new Dimension(175,26));
        viewButton.setPreferredSize(new Dimension(175,26));
        viewButton.setMinimumSize(new Dimension(175,26));
        viewButton.setMaximumSize(new Dimension(175,26));
        viewHButton.setMaximumSize(new Dimension(175,26));
        orig2HButton.setMaximumSize(new Dimension(175,26));

	JPanel pexpress = new JPanel(new SpringLayout());
	JPanel prerepeatv2 = new JPanel();

	p.add(new JLabel(""));
	prerepeatv2.add(viewButton);
	viewButton.addActionListener(this);
	prerepeatv2.add(viewHButton);
	prerepeatv2.add(repeatButton);
        viewHButton.addActionListener(this);
        spotHButton.addActionListener(this);
        repeatButton.addActionListener(this);
        defaultColor = repeatButton.getBackground();
	if (ST.vRepeatFilesDEF.size() >= 1)
	{ 
	   repeatButton.setBackground(ListDialog.buttonColor);
	}

        orig2HButton.addActionListener(this);
        prerepeatv2.setBackground(lightBlue);

	JPanel p15 =new JPanel();
        spotcheck = new JCheckBox(szspotcheck,bspotcheckDEF);
	spotcheck.setBackground(lightBlue);

        p15.add(spotcheck); 
	p15.setBackground(lightBlue);

	prerepeatv2.setBackground(lightBlue);
	prerepeatv2.add(orig2HButton);
	pexpress.add(prerepeatv2);

	lognormButton = new JRadioButton("Log normalize data");
	normButton = new JRadioButton("Normalize data");
	lognormButton.setBackground(lightBlue);
	normButton.setBackground(lightBlue);
 	nonormButton = new JRadioButton("No normalization/add 0");
	nonormButton.setBackground(lightBlue);
	JPanel normPanel = new JPanel();
	if (nnormalizeDEF == 0)
	{
	    lognormButton.setSelected(true);
	}
	else if (nnormalizeDEF == 1)
	{
	    normButton.setSelected(true);
        }
	else 
	{
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
  	p15.add(spotHButton);
	pexpress.add(p15);
	p.add(pexpress);
	pexpress.setBackground(lightBlue);
        SpringUtilities.makeCompactGrid(pexpress,3,1,0,0,0,0);
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
	JLabel step2 = new JLabel("  2.  Gene Info:");

	p.add(step2);
	p.add(new JLabel(""));
	p.add(new JLabel(""));
	p.add(new JLabel(""));

        JPanel psource = new JPanel(new SpringLayout());
	psource.setBackground(lightBlue);

	orgcb = new JComboBox(GoAnnotations.organisms);
	orgcb.setSelectedIndex(ndbDEF);
	orgcb.addActionListener(this);

        goField = new JTextField(szGeneAnnotationFileDEF, JLabel.TRAILING);
        goField.setColumns(NUMCOLS);

	psource.add(new JLabel("Gene Annotation Source: "));
	psource.add(orgcb);
	psource.add(presetsHButton);

        xrefcb =new JComboBox(GoAnnotations.defaultxrefs);
	xrefcb.addActionListener(this);

	psource.add(new JLabel(szxrefsource,JLabel.TRAILING));
	psource.add(xrefcb);
	psource.add(xrefsourceHButton);
	xrefsourceHButton.addActionListener(this);
	handlendbval();
	xrefcb.setSelectedIndex(nxrefDEF);
	handlenxrefval();

	chromcb.addActionListener(this);
	chromcb.setSelectedIndex(nlocationDEF);
	psource.add(new JLabel(szchromsource,JLabel.TRAILING));
	psource.add(chromcb);
	psource.add(chromcbHButton);

	handlechromval();

        SpringUtilities.makeCompactGrid(psource,3,3,0,2,0,2);    
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

	p.add(chromLabel);
	p.add(chromField);
	p.add(chromButton);
	p.add(chromHButton);
        chromButton.addActionListener(this);
        chromHButton.addActionListener(this);
        chromcbHButton.addActionListener(this);

	JPanel pdownload = new JPanel();
	pdownload.setBackground(lightBlue);
	JLabel dlabel =new JLabel("Download:",JLabel.TRAILING);
	pdownload.add(dlabel);

	anncheck.setBackground(lightBlue);
	xrefcheck.setBackground(lightBlue);
	chromcheck.setBackground(lightBlue);
	obocheck.setBackground(lightBlue);
	pdownload.add(anncheck);
	pdownload.add(xrefcheck);
	pdownload.add(chromcheck);
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

	JPanel pclusteringmethod = new JPanel();
	pclusteringmethod.setBackground(lightBlue);
	clusteringmethodcb = new JComboBox(clusteringmethod);
	clusteringmethodcb.addActionListener(this);

	pclusteringmethod.add(new JLabel("Clustering Method:"));
	pclusteringmethod.add(clusteringmethodcb);
	pclusteringmethod.add(clusteringmethodHButton);
	clusteringmethodHButton.addActionListener(this);
	p.add(new JLabel(""));
	p.add(pclusteringmethod);
	p.add(new JLabel(""));
	p.add(new JLabel(""));	

	nK = nKDEF;
	nreps = nREPDEF;
        nMaxProfiles = nMaxProfilesDEF;
        nMaxChangeProfiles = nMaxUnitDEF;
        SpinnerNumberModel snmodelProfile; 
        SpinnerNumberModel snmodelChange; 

	if (nCLUSTERINGMETHODDEF == 0) 
	{
           snmodelProfile = new SpinnerNumberModel(new Integer(nMaxProfilesDEF),new Integer(0),
                                                                     null,new Integer(5));
           snmodelChange = new SpinnerNumberModel(new Integer(nMaxUnitDEF),new Integer(1),
                                                                          null,new Integer(1));
	}
	else
	{
           snmodelProfile = new SpinnerNumberModel(new Integer(nK),new Integer(1),
                                                                     null,new Integer(1));
           snmodelChange = new SpinnerNumberModel(new Integer(nreps),new Integer(1),
                                                                          null,new Integer(1));

	    maxchangeLabel.setText(szrep);
	    numberprofilesLabel.setText(szk);
	}
	numberprofilesLabel.setMinimumSize(new Dimension(356,26));
        numberprofilesLabel.setMaximumSize(new Dimension(356,26));

        thespinnerProfile = new JSpinner(snmodelProfile);
        thespinnerProfile.setPreferredSize(new Dimension(60,24));
        thespinnerProfile.setMinimumSize(new Dimension(60,24));
        thespinnerProfile.setMaximumSize(new Dimension(60,24));  

        SpinnerNumberModel snmodelmingo = new SpinnerNumberModel(new Integer(nMinGoGenesDEF),new Integer(1),
                                                                           null,new Integer(1));
        thespinnermingo = new JSpinner(snmodelmingo);
        thespinnermingo.setPreferredSize(new Dimension(60,24));
        thespinnermingo.setMaximumSize(new Dimension(60,24));

        SpinnerNumberModel snmodelmingolevel = new SpinnerNumberModel(new Integer(nMinGOLevelDEF),new Integer(1),
                                                                         null,new Integer(1));
        thespinnermingolevel = new JSpinner(snmodelmingolevel);
        thespinnermingolevel.setPreferredSize(new Dimension(60,24));
        thespinnermingolevel.setMaximumSize(new Dimension(60,24));

        thespinnerChange = new JSpinner(snmodelChange);
        thespinnerChange.setPreferredSize(new Dimension(60,24));
        thespinnerChange.setMaximumSize(new Dimension(60,24));
        thespinnerChange.setMinimumSize(new Dimension(60,24));

        SpinnerNumberModel snmodelMissing = new SpinnerNumberModel(new Integer(nMaxMissingDEF), new Integer(0),
                                                                           null, new Integer(1));
        thespinnermaxmissing = new JSpinner(snmodelMissing); 
        thespinnermaxmissing.setPreferredSize(new Dimension(60,24));
        thespinnermaxmissing.setMaximumSize(new Dimension(60,24));
        SpinnerNumberModel snmodelFilter = new SpinnerNumberModel(new Double(dMinCorrelationRepeatsDEF), 
                                                                  new Double(-1.1),
                                                                  new Double(1.1), new Double(0.05));
        thespinnerfilterthreshold = new JSpinner(snmodelFilter); 
        thespinnerfilterthreshold.setPreferredSize(new Dimension(60,24));
        thespinnerfilterthreshold.setMaximumSize(new Dimension(60,24));

        SpinnerNumberModel snmodelExpress = new SpinnerNumberModel(new Double(dMinExpressionDEF), 
                                          new Double(-.05),null, new Double(0.05));
        thespinnerexpress = new JSpinner(snmodelExpress); 
        thespinnerexpress.setPreferredSize(new Dimension(60,24));
        thespinnerexpress.setMaximumSize(new Dimension(60,24));

        SpinnerNumberModel snmodelCorr = new SpinnerNumberModel(new Double(dMaxCorrelationModelDEF), new Double(-1),
                                                                new Double(1), new Double(0.05));
        thespinnercorrmodel = new JSpinner(snmodelCorr); 
        thespinnercorrmodel.setPreferredSize(new Dimension(60,24));
        thespinnercorrmodel.setMaximumSize(new Dimension(60,24));

        SpinnerNumberModel snmodelAlpha = new SpinnerNumberModel(new Double(dSignificanceLevelDEF), new Double(0),
                                                                 null, new Double(0.005));
        thespinneralpha = new JSpinner(snmodelAlpha); 
        thespinneralpha.setPreferredSize(new Dimension(60,24));
        thespinneralpha.setMaximumSize(new Dimension(60,24));

        SpinnerNumberModel snmodelLB = new SpinnerNumberModel(new Double(dMinimumCorrelationClusteringDEF), 
                                                                   new Double(-1), new Double(1), new Double(0.05));
        thespinnerlb = new JSpinner(snmodelLB); 
        thespinnerlb.setPreferredSize(new Dimension(60,24));
        thespinnerlb.setMaximumSize(new Dimension(60,24));


        SpinnerNumberModel snmodelPercentile = new SpinnerNumberModel(new Double(dMinimumPercentileClusteringDEF), 
                                                                   new Double(0), new Double(1), new Double(0.05));
        thespinnerpercentile = new JSpinner(snmodelPercentile); 
        thespinnerpercentile.setPreferredSize(new Dimension(60,24));
        thespinnerpercentile.setMaximumSize(new Dimension(60,24));

        SpinnerNumberModel snmodelGene = new SpinnerNumberModel(new Integer(nNumPermsGeneDEF), new Integer(0),
                                                                   null, new Integer(5));
        thespinnersamplegene = new JSpinner(snmodelGene); 
        thespinnersamplegene.setPreferredSize(new Dimension(60,24));
        thespinnersamplegene.setMaximumSize(new Dimension(60,24));

        SpinnerNumberModel snmodelModel = new SpinnerNumberModel(new Long(nMaxCandidateModelDEF), new Long(1),
                                                                   null, new Long(1000));
        thespinnersamplemodel = new JSpinner(snmodelModel); 
        thespinnersamplemodel.setPreferredSize(new Dimension(100,24));
        thespinnersamplemodel.setMaximumSize(new Dimension(100,24));

        SpinnerNumberModel snmodelpval = new SpinnerNumberModel(new Integer(nSamplesMultipleDEF), new Integer(1),
                                                                  null, new Integer(50));
        thespinnersamplepval = new JSpinner(snmodelpval); 
        thespinnersamplepval.setPreferredSize(new Dimension(60,24));
        thespinnersamplepval.setMaximumSize(new Dimension(60,24));

	xrefLabel.setLabelFor(xrefField);
        orig1Label.setLabelFor(orig1Field);
        extraLabel.setLabelFor(extraField);
	categoryIDLabel.setLabelFor(categoryIDField);
        alphaLabel.setLabelFor(thespinneralpha);
        lbLabel.setLabelFor(thespinnerlb);
        maxmissingLabel.setLabelFor(thespinnermaxmissing);
        filterthresholdLabel.setLabelFor(thespinnerfilterthreshold);
        expressLabel.setLabelFor(thespinnerexpress);
        maxchangeLabel.setLabelFor(thespinnerChange);
        numberprofilesLabel.setLabelFor(thespinnerProfile);
        corrmodelLabel.setLabelFor(thespinnercorrmodel);
        percentileLabel.setLabelFor(thespinnerpercentile);
        samplegeneLabel.setLabelFor(thespinnersamplegene);
        samplemodelLabel.setLabelFor(thespinnersamplemodel);
        samplepvalLabel.setLabelFor(thespinnersamplepval);
        goLabel.setLabelFor(goField);
        mingoLabel.setLabelFor(thespinnermingo);
        mingolevelLabel.setLabelFor(thespinnermingolevel);

        pcluster = new JPanel(new SpringLayout());
        pcluster.setBackground(lightBlue);
        pcluster.add(numberprofilesLabel);
        pcluster.add(thespinnerProfile);        
        pcluster.add(numberProfileHButton);
        numberProfileHButton.addActionListener(this);
        pcluster.add(maxchangeLabel);
        pcluster.add(thespinnerChange);
        pcluster.add(changeHButton);
        changeHButton.addActionListener(this);
        SpringUtilities.makeCompactGrid(pcluster,2,3,20,0,3,2);  
	clusteringmethodcb.setSelectedIndex(nCLUSTERINGMETHODDEF);  

	p.add(new JLabel(""));
	p.add(pcluster);
	p.add(new JLabel(""));
        p.add(new JLabel(""));

        JPanel padvanced = new JPanel();
        clusterAButton = new JButton(szClusterA);
        clusterAButton.setBackground(ST.buttonColor);
        clusterAButton.addActionListener(this); 
        clusterAButton.setPreferredSize(new Dimension(200,26));
        clusterAButton.setMinimumSize(new Dimension(200,26));
        clusterAButton.setMaximumSize(new Dimension(200,26));
  

        optionsButton.setPreferredSize(new Dimension(250,28));
        optionsButton.setMinimumSize(new Dimension(250,28));
        optionsButton.setMaximumSize(new Dimension(250,28));

        padvanced.add(optionsButton);
        padvanced.add(advancedHButton);
	padvanced.setBackground(lightBlue);

        advancedHButton.addActionListener(this);
	p.add(new JLabel(""));
        p.add(padvanced);
	p.add(new JLabel(""));
	p.add(new JLabel(""));
  
        optionsButton.addActionListener(this);
        contentPane.add(Box.createRigidArea(new Dimension(0,6)));
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
	cpanel.add(openButton);
	cpanel.setBackground(lightBlue);
	cpanel.add(clusterAButton);
	cpanel.add(executeHButton);
        p.add(cpanel);
	p.add(new JLabel(""));
	p.add(new JLabel(""));

        SpringUtilities.makeCompactGrid(p,18,4,0,2,0,0);        
        contentPane.add(p);

        char ch = (char) 169;
        JLabel copyrightLabel = new JLabel(ch+" 2004, Carnegie Mellon University.  All Rights Reserved.");
        copyrightLabel.setAlignmentX(Component.CENTER_ALIGNMENT);
        JPanel infopanel = new JPanel(new SpringLayout());

        infopanel.add(copyrightLabel);
        infopanel.add(infoButton);
	openButton.addActionListener(this);
        infoButton.addActionListener(this);
	infoButton.setBackground(ST.lightBlue);
        SpringUtilities.makeCompactGrid(infopanel,1,2,0,4,0,0);     
 
        infopanel.setBackground(lightBlue);
        contentPane.add(infopanel);
        contentPane.add(Box.createRigidArea(new Dimension(0,2)));

        makeOptionsDialog();

        theRepeatList =  new ListDialog(this,
					ST.vRepeatFilesDEF , ST.balltimeDEF,
                                        repeatButton,ST.lightBlue,ST.defaultColor,ST.fc);
    }

    /**
     * Returns true iff sztype is a valid variable for the defaults file
     */
    public static boolean validVariable(String sztype)
    {

        return (
	      (sztype.equalsIgnoreCase("Show_Main_Y-axis_gene_tick_marks"))||
	      (sztype.equalsIgnoreCase("Display_Profile_ID"))||
	      (sztype.equalsIgnoreCase("Display_K-Means_Cluster_ID"))||
	      (sztype.equalsIgnoreCase("Display_Model_Profile"))||
	      (sztype.equalsIgnoreCase("Display_Cluster_Mean"))||
	      (sztype.equalsIgnoreCase("Display_details_when_ordering"))||
	      (sztype.equalsIgnoreCase("Main_Y-axis_gene_tick_interval"))||
              (sztype.equalsIgnoreCase("Scale_should_be_based_on_only_selected_genes"))||
              (sztype.equalsIgnoreCase("Y_Scale_Max"))||
	      (sztype.equalsIgnoreCase("Max"))||
              (sztype.equalsIgnoreCase("Y_Scale_Min"))||
	      (sztype.equalsIgnoreCase("Min"))||
              (sztype.equalsIgnoreCase("X-axis_scale_should_be[Uniform,Based on real time]"))||
	      (sztype.equalsIgnoreCase("X-axis_scale_should_be"))||
	      (sztype.equalsIgnoreCase("Y-axis_scale_on_details_windows_should_be"))||
	      (sztype.equalsIgnoreCase("Y-axis_scale_on_details_windows_should_be[Determined automatically,Fixed]"))||
	       (sztype.equalsIgnoreCase("Y-axis_scale_on_details_windows_should_be[Determined automatically,Fixed with parameters below]"))||
               (sztype.equalsIgnoreCase("Tick_interval"))||
              (sztype.equalsIgnoreCase("Y-axis_scale_for_genes_on_main_interface_should_be"))||
	       (sztype.equalsIgnoreCase("Y-axis_scale_for_genes_on_main_interface_should_be[Gene specific,Profile specific,Global]"))||
            (sztype.equalsIgnoreCase("Gene_Color(R,G,B)"))||
            (sztype.equalsIgnoreCase("Gene_Color"))||
             (sztype.equalsIgnoreCase("Y-axis_scale_for_genes_on_k-means_main_interface_should_be"))||
	     (sztype.equalsIgnoreCase("Y-axis_scale_for_genes_on_k-means_main_interface_should_be[Cluster specific,Global]"))||
            (sztype.equalsIgnoreCase("Gene_display_policy_on_main_interface"))||
            (sztype.equalsIgnoreCase("Gene_display_policy_on_main_interface[Do not display,Display only selected,Display all]"))||
            (sztype.equalsIgnoreCase("Normalize_Data"))||
            (sztype.equalsIgnoreCase("Normalize_Data[Log normalize data,Normalize data,No normalization/add 0]"))||
	    (sztype.equalsIgnoreCase("Change_should_be_based_on[Maximum-Minimum,Difference From 0]"))||
            (sztype.equalsIgnoreCase("Gene_Annotation_Source"))||
            (sztype.equalsIgnoreCase("Include_Biological_Process"))||
            (sztype.equalsIgnoreCase("Include_Molecular_Function"))||
            (sztype.equalsIgnoreCase("Include_Cellular_Process"))||
            (sztype.equalsIgnoreCase("Permutation_Test_Should_Permute_Time_Point_0"))||
            (sztype.equalsIgnoreCase("Only_include_annotations_with_these_evidence_codes"))||
            (sztype.equalsIgnoreCase("Only_include_annotations_with_these_taxon_IDs"))||
            (sztype.equalsIgnoreCase("Cross_Reference_Source"))||
            (sztype.equalsIgnoreCase("Cross_Reference_File"))||
            (sztype.equalsIgnoreCase("Gene_Location_Source"))||
            (sztype.equalsIgnoreCase("Gene_Location_File"))||
            (sztype.equalsIgnoreCase("Data_File"))||
            (sztype.equalsIgnoreCase("Category_ID_File"))||
            (sztype.equalsIgnoreCase("Gene_Annotation_File"))||
            (sztype.equalsIgnoreCase("Spot_IDs_included_in_the_the_data_file"))||
            (sztype.equalsIgnoreCase("Spot_IDs_included_in_the_data_file"))||
            (sztype.equalsIgnoreCase("Spot_IDs_in_the_data_file"))||
            (sztype.equalsIgnoreCase("Comparison_Data_File"))||
            (sztype.equalsIgnoreCase("Comparison_Repeat_Data_Files(comma delimited list)"))||
            (sztype.equalsIgnoreCase("Comparison_Repeat_Data_Files"))||
            (sztype.equalsIgnoreCase("Comparison_Repeat_Data_is_from"))||
            (sztype.equalsIgnoreCase("Comparison_Repeat_Data_is_from[Different time periods,The same time period]"))||
            (sztype.equalsIgnoreCase("Comparison_Minimum_Number_of_genes_in_intersection"))||
            (sztype.equalsIgnoreCase("Comparison_Maximum_Uncorrected_Intersection_pvalue"))||
            (sztype.equalsIgnoreCase("Maximum_Number_of_Model_Profiles"))||
	    (sztype.equalsIgnoreCase("Clustering_Method[STEM Clustering Method,K-means]"))||
            (sztype.equalsIgnoreCase("Clustering_Method"))||
            (sztype.equalsIgnoreCase("Number_of_Clusters"))||
            (sztype.equalsIgnoreCase("Number_of_Clusters_K"))||
            (sztype.equalsIgnoreCase("Number_of_Random_Starts"))||
            (sztype.equalsIgnoreCase("Maximum_Unit_Change_in_Model_Profiles_between_Time_Points"))||
            (sztype.equalsIgnoreCase("Repeat_Data_Files(comma delimited list)"))||
            (sztype.equalsIgnoreCase("Repeat_Data_Files"))||
            (sztype.equalsIgnoreCase("Repeat_Data_is_from"))||
            (sztype.equalsIgnoreCase("Repeat_Data_is_from[Different time periods,The same time period]"))||
            (sztype.equalsIgnoreCase("Maximum_Number_of_Missing_Values"))||
            (sztype.equalsIgnoreCase("Minimum_Absolute_Log_Ratio_Expression"))||
            (sztype.equalsIgnoreCase("Minimum_Correlation_between_Repeats"))||
            (sztype.equalsIgnoreCase("Pre-filtered_Gene_File"))||
            (sztype.equalsIgnoreCase("Maximum_Correlation"))||
            (sztype.equalsIgnoreCase("Number_of_Permutations_per_Gene"))||
            (sztype.equalsIgnoreCase("Maximum_Number_of_Candidate_Model_Profiles"))||
            (sztype.equalsIgnoreCase("Significance_Level"))||
            (sztype.equalsIgnoreCase("Multiple_hypothesis_correction_method_enrichment[Bonferroni,Randomization]"))||
            (sztype.equalsIgnoreCase("Correction_Method[Bonferroni,False Discovery Rate,None]"))||
            (sztype.equalsIgnoreCase("Clustering_Minimum_Correlation"))||
            (sztype.equalsIgnoreCase("Clustering_Minimum_Correlation_Percentile"))||
            (sztype.equalsIgnoreCase("Number_of_samples_for_randomized_multiple_hypothesis_correction"))||
            (sztype.equalsIgnoreCase("GO_Minimum_number_of_genes"))||
            (sztype.equalsIgnoreCase("Minimum_GO_level")));
    }        
      

    /**
     * Initializes the default settings of all the variables
     */
    public static void initializeDefaults()
    {
       szCompareDEF = "";
       vRepeatCompareDEF = new Vector();
       nCompareMinGenesDEF = 5;
       dCompareMinpvalDEF = 0.005;
       bcomparealltimeDEF = true;
       sztaxonDEF = "";
       szevidenceDEF = "";
       szDefaultFile = "";
       szDataFileDEF = "";
       szCrossRefFileDEF = "";
       szGeneAnnotationFileDEF = "";
       szChromFileDEF = "";
       szGeneOntologyFileDEF = "gene_ontology.obo";
       nMaxProfilesDEF = 50;
       ngenedisplayDEF = 0;
       ngenescaleDEF = 1;
       ngenescalekmeansDEF = 2;
       dyscaleminDEF = -3;
       dyscalemaxDEF = 3;
       dtickDEF = 1;
       bautomaticYDEF = true;
       buniformXaxisDEF = true;
       bscalevisibleDEF = true;
       bdisplayIDDEF = true; 
       bdisplayprofilelineDEF = true; 
       bdisplaymaintickDEF = false; 
       bdisplaydetailDEF = true; 
       dtickmainDEF = 1;
       nKDEF = 10;
       nREPDEF = 20;
       nCLUSTERINGMETHODDEF = 0;
       nMaxUnitDEF = 2;
       vRepeatFilesDEF = new Vector();
       balltimeDEF = true;
       nMaxMissingDEF = 0;
       genecolorRDEF = 204;
       genecolorGDEF = 51;
       genecolorBDEF = 0;
       dMinExpressionDEF = 1;
       dMinCorrelationRepeatsDEF = 0;
       szPrefilteredDEF = "";
       dMaxCorrelationModelDEF = 1;
       nNumPermsGeneDEF = 50;
       nMaxCandidateModelDEF = 1000000;
       dSignificanceLevelDEF = .05;
       nfdrDEF = 2;
       dMinimumCorrelationClusteringDEF = .7; 
       dMinimumPercentileClusteringDEF = 0; 
       nSamplesMultipleDEF = 500;
       nMinGoGenesDEF = 5;
       nMinGOLevelDEF = 3; 
       ndbDEF = 0;
       nxrefDEF = 0;
       nnormalizeDEF = 1;
       nlocationDEF = 0;
       bspotcheckDEF = false;
       szxrefDEF = "";
       szcategoryIDDEF = "";
       szlocationDEF = "";
       bpontoDEF = true;
       bcontoDEF = true;
       bfontoDEF = true;
       brandomgoDEF = true;
       bmaxminDEF = true;
       ballpermuteDEF =true;
    }

    /**
     * Runs STEM on a batch set of default files in szBatchInputDir
     */
    public void runBatchDir()
    {
       File dir = new File(szBatchInputDir);
       if (!dir.exists())
       {
          System.out.println("WARNING: "+szBatchInputDir+" does not exist");
       }
       else 
       {
          boolean binputdir;
          String[] children;
	  if (dir.isDirectory())
          {
	     binputdir = true;
	     children = dir.list();
	  }
          else
	  {
	      binputdir = false;
	      children = new String[1];
	  }

	  for (int nindex = 0; nindex < children.length; nindex++)
	  {
	     boolean bexecuteok = true;
             try
             {
	        ST.initializeDefaults();
	        if (binputdir)
	        {
	           System.out.println("Processing "+children[nindex]);
	           szcurrentDefaultFile = children[nindex];
	           parseDefaults(szBatchInputDir+File.separator+children[nindex]);
	        }
	        else
	        {
	           szcurrentDefaultFile = szBatchInputDir;
	           parseDefaults(szBatchInputDir);
	        }
	     }
	     catch (IllegalArgumentException ex)
	     {
	        System.out.println(ex.getMessage());
	     }
	     catch (IOException ex)
	     {
	        bexecuteok = false;
	        ex.printStackTrace(System.out);
	     }
	    
	     if (bexecuteok)
	     {
                try
	        {
	           String sznumclusterORprofile;
		   String sznumrepsORunitchange;
		   nclusteringmethodcb = nCLUSTERINGMETHODDEF;
	           if (nCLUSTERINGMETHODDEF == 0)
	           {
	              //stem method
		       sznumclusterORprofile = ""+nMaxProfilesDEF;
		       sznumrepsORunitchange = ""+nMaxUnitDEF;
		   }
	           else
	           {
	              //we got k-means
		       sznumclusterORprofile = ""+nKDEF;
		       sznumrepsORunitchange = ""+nREPDEF;
		    }

		    clusterscript(szChromFileDEF,szCrossRefFileDEF,szDataFileDEF,
                               szorig2val,szGeneAnnotationFileDEF,
                               szgocategoryval, ""+nMaxMissingDEF,
                               ""+dMinExpressionDEF,""+dMinCorrelationRepeatsDEF,
                               ""+dMinimumCorrelationClusteringDEF, ""+dSignificanceLevelDEF,
                               ""+dMinimumPercentileClusteringDEF, ""+sznumrepsORunitchange,
                               sznumclusterORprofile, ""+dMaxCorrelationModelDEF, 
                               ""+nNumPermsGeneDEF,""+nSamplesMultipleDEF,
                               ""+nMaxCandidateModelDEF, ""+nMinGoGenesDEF,nfdrDEF,""+nMinGOLevelDEF,
			       szPrefilteredDEF,balltimeDEF,vRepeatFilesDEF,(nnormalizeDEF==0),//btakelog,
			       false,false,false,bspotcheckDEF,(nnormalizeDEF==2),
                               szcategoryIDDEF, 
                               //does not download new files in batch mode
			       szevidenceDEF,sztaxonDEF,bpontoDEF,
			       bcontoDEF,bfontoDEF,brandomgoDEF,bmaxminDEF,ballpermuteDEF);


		}
	        catch(Exception ex)
	        {
	           ex.printStackTrace(System.out);
	        }
	     }
	  }
       }
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Parse the contents of a defaults file
     */
    public static void parseDefaults(String szDefaultFile) throws FileNotFoundException, IOException
    {
       String szLine;
       BufferedReader br;
       int ntempval;
       double dtempval;

       vRepeatCompareDEF = new Vector();
       vRepeatFilesDEF = new Vector();

       String szError = "";
       br = new BufferedReader(new FileReader(szDefaultFile));
       while ((szLine = br.readLine())!= null)
       { 
          StringTokenizer st = new StringTokenizer(szLine,"\t");
          if (st.hasMoreTokens())
	  {
	     String sztype = st.nextToken().trim();
             if (sztype.charAt(0)!='#')
	     {
	        String szvalue = "";
                if (st.hasMoreTokens())
		{
	           szvalue = st.nextToken().trim();
		} 

		if (!validVariable(sztype))
	        {
	           szError+= "WARNING: '"+sztype+"' is an unrecognized variable.\n";
	        }
	        else if (!szvalue.equals(""))
	        {
	           if ((sztype.equalsIgnoreCase("Gene_display_policy_on_main_interface[Do not display,Display only selected,Display all]"))||
		       (sztype.equalsIgnoreCase("Gene_display_policy_on_main_interface")))
		   {
                      try
		      {
	                 ngenedisplayDEF = Integer.parseInt(szvalue);
		         if ((ngenedisplayDEF < 0)|| (ngenedisplayDEF >2))
		         {
		            szError +=szvalue+" is an invalid argument for Gene_display_policy_on_main_interface\n";
		         }
	              }
		      catch(NumberFormatException ex)
	              {
	                 if (szvalue.equalsIgnoreCase("Do not display"))
	                 {
                            ngenedisplayDEF = 0;
	                 }
	                 else if (szvalue.equalsIgnoreCase("Display only selected"))
	                 {
		             ngenedisplayDEF = 1;
	                 } 
		         else if (szvalue.equalsIgnoreCase("Display all"))
	       	         {
	                    ngenedisplayDEF = 2;
	                 }
		         else
		         {
	                    szError +=szvalue+
                               " is an invalid argument for Gene_display_policy_on_main_interface\n";
      	                 }
	              }
		   }
	           else if (sztype.equalsIgnoreCase("Show_Main_Y-axis_gene_tick_marks"))
		   {
		      if (szvalue.equalsIgnoreCase("true"))
		      {
		         ST.bdisplaymaintickDEF =  true;
		      }
		      else if (szvalue.equalsIgnoreCase("false"))
		      {
			 ST.bdisplaymaintickDEF =  false;
		      }
		      else
		      {
                         szError += "Warning: "+szvalue +" is an unrecognized "+
				"value for Show_Main_Y-axis_gene_tick_marks (expecting true or false)\n";
		      }
		   }
		   else if (sztype.equalsIgnoreCase("Main_Y-axis_gene_tick_interval"))
		   {
		      dtempval = Double.parseDouble(szvalue);
		      if (dtempval <= 0)
		      {
		         szError +="WARNING: '"+szvalue+
                               "' is an invalid value for Main_Y-axis_gene_tick_interval "+
                               "must be > 0\n";
		      }
		      else
	              {
			  dtickmainDEF = dtempval;
		      }
		   }
		   else if ((sztype.equalsIgnoreCase("Display_Profile_ID"))||
			    (sztype.equalsIgnoreCase("Display_K-Means_Cluster_ID")))
		   {
		      if (szvalue.equalsIgnoreCase("true"))
		      {
		         ST.bdisplayIDDEF =  true;
		      }
		      else if (szvalue.equalsIgnoreCase("false"))
		      {
			 ST.bdisplayIDDEF =  false;
		      }
		      else
		      {
                         szError += "Warning: "+szvalue +" is an unrecognized "+
				 "value for "+sztype+" (expecting true or false)\n";
		      }
		   }
		   else if ((sztype.equalsIgnoreCase("Display_Model_Profile"))||
				 (sztype.equalsIgnoreCase("Display_Cluster_Mean")))
		   {
		      if (szvalue.equalsIgnoreCase("true"))
		      {
		         ST.bdisplayprofilelineDEF = true;
		      }
		      else if (szvalue.equalsIgnoreCase("false"))
		      {
		         ST.bdisplayprofilelineDEF = false;
		      }
		      else
		      {
                         szError += "Warning: "+szvalue +" is an unrecognized "+
				 "value for "+sztype+" (expecting true or false)\n";
		      }
		   }
		   else if (sztype.equalsIgnoreCase("Display_details_when_ordering"))
		   {
		      if (szvalue.equalsIgnoreCase("true"))
		      {
		         ST.bdisplaydetailDEF = true;
		      }
		      else if (szvalue.equalsIgnoreCase("false"))
		      {
		         ST.bdisplaydetailDEF = false;
		      }
		      else
		      {
                         szError += "Warning: "+szvalue +" is an unrecognized "+
				 "value for Display_details_when_ordering (expecting true or false)\n";
		      }
		   }
		   else if ((sztype.equalsIgnoreCase("Gene_Color(R,G,B)"))||
                                  (sztype.equalsIgnoreCase("Gene_Color")))
		   {
                      StringTokenizer stColors = new StringTokenizer(szvalue, ",");
		      if (stColors.countTokens() != 3)
		      {
			  szError +=szvalue+" is an invalid R,G,B value\n";
		      }
		      else
		      {
                         try
		         {
			     genecolorRDEF = Integer.parseInt(stColors.nextToken());
			     genecolorGDEF = Integer.parseInt(stColors.nextToken());
			     genecolorBDEF = Integer.parseInt(stColors.nextToken());
			 }
		         catch(NumberFormatException ex)
		         {
			     szError +=szvalue+" is an invalid R,G,B value\n";
			 }
		      }
		   }
		   else if (
                    (sztype.equalsIgnoreCase("Y-axis_scale_on_details_windows_should_be"))||
	            (sztype.equalsIgnoreCase("Y-axis_scale_on_details_windows_should_be[Determined automatically,Fixed]"))||
		    (sztype.equalsIgnoreCase("Y-axis_scale_on_details_windows_should_be[Determined automatically,Fixed with parameters below]")))
		   {
		      if (szvalue.equalsIgnoreCase("Determined automatically"))
		      {
		         bautomaticYDEF =  true;
		      }
		      else if (szvalue.equalsIgnoreCase("Fixed"))
		      {
	                 bautomaticYDEF =  false;
		      }
		      else
		      {
                         szError += "Warning: "+szvalue +" is an unrecognized "+
				  "value for Y-axis_scale_on_details_windows_should_be\n";
	      	      }
		   }
		   else if ((sztype.equalsIgnoreCase("X-axis_scale_should_be[Uniform,Based on real time]"))||
				  (sztype.equalsIgnoreCase("X-axis_scale_should_be"))||
			          (sztype.equalsIgnoreCase("X-axis_scale[Uniform,Based on Real Time]"))||
				  (sztype.equalsIgnoreCase("X-axis_scale")))
		   {
		      if (szvalue.equalsIgnoreCase("Uniform"))
		      {
			  buniformXaxisDEF =  true;
		      }
		      else if (szvalue.equalsIgnoreCase("Based on real time"))
		      {
			  buniformXaxisDEF =  false;
		      }
		      else
		      {
                         szError += "Warning: "+szvalue +" is an unrecognized "+
				  "value for X-axis_scale_should_be\n";
		      }
		   }
		   else if ((sztype.equalsIgnoreCase("Y-axis_scale_for_genes_on_main_interface_should_be"))||
		     (sztype.equalsIgnoreCase("Y-axis_scale_for_genes_on_main_interface_should_be[Gene specific,Profile specific,Global]")))
		   {
                      try
		      {
			  ngenescaleDEF = Integer.parseInt(szvalue);
		          if ((ngenescaleDEF < 0)|| (ngenescaleDEF >2))
		          {
		             szError +=szvalue+" is an invalid argument for Y-axis_scale_for_genes_on_main_interface_should_be\n";
		          }
		      }
		      catch(NumberFormatException ex)
	              {
		         if (szvalue.equalsIgnoreCase("Gene specific"))
		         {
			     ngenescaleDEF = 0;
			 }
		         else if (szvalue.equalsIgnoreCase("Profile specific"))
		         {
			     ngenescaleDEF = 1;
			 }
		         else if (szvalue.equalsIgnoreCase("Global"))
		         {
			     ngenescaleDEF = 2;
			 }
		         else
		         {
		            szError +=szvalue+
                                   " is an invalid argument for Y-axis_scale_for_genes_on_main_interface_should_be\n";
			 }
		      }
		   }
		   else if ((sztype.equalsIgnoreCase("Y-axis_scale_for_genes_on_k-means_main_interface_should_be"))||
			    (sztype.equalsIgnoreCase("Y-axis_scale_for_genes_on_k-means_main_interface_should_be[Cluster specific,Global]")))
		   {
                      try
		      {
		         ngenescalekmeansDEF = Integer.parseInt(szvalue);
		         if ((ngenescaleDEF < 0)|| (ngenescaleDEF >2))
		         {
			     szError +=szvalue+" is an invalid argument for Y-axis_scale_for_genes_on_k-means_main_interface_should_be\n";
			 }
		      }
		      catch(NumberFormatException ex)
		      {
                         if (szvalue.equalsIgnoreCase("Cluster specific"))
		         {
			    ngenescalekmeansDEF = 1;
		         } 
			 else if (szvalue.equalsIgnoreCase("Global"))
			 {
		            ngenescalekmeansDEF = 2;
		         }
		         else
		         {
		            szError +=szvalue+
                                   " is an invalid argument for Y-axis_scale_for_genes_on_k-means_main_interface_should_be\n";
			 }
		      }
		   }
		   else if ((sztype.equalsIgnoreCase("Min"))||
			    (sztype.equalsIgnoreCase("Y_Scale_Min")))
		   {
		       dtempval = Double.parseDouble(szvalue);
		       if (dtempval > 0)
		       {
		          szError +="WARNING: '"+szvalue+
                               "' is an invalid value for Y_Scale_Min, must be <0\n";
		       }
		       else
		       {
		          dyscaleminDEF = dtempval;
		       }
	            }
		    else if ((sztype.equalsIgnoreCase("Y_Scale_Max"))||
			     (sztype.equalsIgnoreCase("Max")))
		    {
                       dtempval = Double.parseDouble(szvalue);
		       if (dtempval < 0)
		       {
		          szError +="WARNING: '"+szvalue+
                               "' is an invalid value for Y_Scale_Max, must be >0\n";
		       }
		       else
		       {
			   dyscalemaxDEF = dtempval;
		       }
		    }
		    else if (sztype.equalsIgnoreCase("Tick_interval"))
		    {
			dtempval = Double.parseDouble(szvalue);
			if (dtempval <= 0)
			{
			   szError +="WARNING: '"+szvalue+
                               "' is an invalid value for Tick interval\n";
			}
		        else
		        {
			    dtickDEF = dtempval;
			}
		    }
		    else if (sztype.equalsIgnoreCase("Scale_should_be_based_on_only_selected_genes"))
		    {
		       if (szvalue.equalsIgnoreCase("true"))
		       {
			   ST.bscalevisibleDEF =  true;
		       }
		       else if (szvalue.equalsIgnoreCase("false"))
		       {
		           ST.bscalevisibleDEF =  false;
		       }
		       else
		       {
                           szError += "Warning: "+szvalue +" is an unrecognized "+
				 "value for Scale_should_be_based_on_only_selected_genes (expecting true or false)\n";
		       }
		    }
		    else if ((sztype.equalsIgnoreCase("Normalize_Data"))||
                    (sztype.equalsIgnoreCase("Normalize_Data[Log normalize data,Normalize data,No normalization/add 0]")))
		    {
                       try
	               {
		          nnormalizeDEF = Integer.parseInt(szvalue);
	                  if ((nnormalizeDEF < 0)|| (nnormalizeDEF >2))
	                  {
		             szError +=szvalue+" is an invalid argument for Normalize_Data\n";
			  }
		       }
		       catch(NumberFormatException ex)
	               {
		          if (szvalue.equalsIgnoreCase("Log normalize data"))
		          {
                             nnormalizeDEF = 0;
	                  }
		          else if (szvalue.equalsIgnoreCase("Normalize data"))
		          {
		             nnormalizeDEF = 1;
	                  } 
		          else if (szvalue.equalsIgnoreCase("No normalization/add 0"))
		          {
	                     nnormalizeDEF = 2;
	                  }
		          else
		          {
			     szError +=szvalue+" is an invalid argument for Normalize_Data\n";
			  }
		       }
		    }  
		    else if (sztype.equalsIgnoreCase("Change_should_be_based_on[Maximum-Minimum,Difference From 0]"))
		    {
		       if (szvalue.equalsIgnoreCase("Maximum-Minimum"))
		       {
		          bmaxminDEF = true;
		       }
		       else if (szvalue.equalsIgnoreCase("Difference From 0"))
		       {
		          bmaxminDEF = false;
		       }
		       else
		       {
		          szError += szvalue+" is an invalid value of "+
                                "Change_should_be_based_on[Maximum-Minimum,Difference From 0]\n";
		        }
		    }  
                    else if (sztype.equalsIgnoreCase("Gene_Annotation_Source"))
		    {
                       try
		       {
		          ndbDEF = Integer.parseInt(szvalue);
			  if ((ndbDEF < 0)|| (ndbDEF >= GoAnnotations.organisms.length))
			  {
			      ndbDEF = 0;
			  }
		       }
		       catch(NumberFormatException ex)
		       {
		          boolean bfound = false;
			  int nsource = 0;
		          while ((nsource < GoAnnotations.organisms.length)&&(!bfound))
		          {
			     if (GoAnnotations.organisms[nsource].equalsIgnoreCase(szvalue))
			     {
			        bfound = true;
			        ndbDEF = nsource;
			     }
  			     else
		             {
			        nsource++;
			     }
			  }     

			  if (!bfound)
		          {
	                     szError += "Warning: "+szvalue +" is an unrecognized "+
		  		        "type for Gene Annotation Source\n";
			  }
		       }
		    }
                    else if (sztype.equalsIgnoreCase("Include_Biological_Process"))
	            {
		       if (szvalue.equalsIgnoreCase("true"))
		       {
		           bpontoDEF =  true;
		       }
		       else if (szvalue.equalsIgnoreCase("false"))
		       {
		           bpontoDEF =  false;
		       }
		       else
		       {
                           szError += "Warning: "+szvalue +" is an unrecognized "+
		      	       "value for Include_Biological_Process (expecting true or false)\n";
		       }
		    }
                    else if (sztype.equalsIgnoreCase("Include_Molecular_Function"))
	            {
		       if (szvalue.equalsIgnoreCase("true"))
		       {
		          bfontoDEF =  true;
		       }
		       else if (szvalue.equalsIgnoreCase("false"))
		       {
		          bfontoDEF =  false;
		       }
		       else
		       {
                          szError += "Warning: "+szvalue +" is an unrecognized "+
	         		 "value for Include_Molecular_Function (expecting true or false)\n";
		        }
                     }
                     else if (sztype.equalsIgnoreCase("Include_Cellular_Process"))
	             {
		        if (szvalue.equalsIgnoreCase("true"))
		        {
 		           bcontoDEF =  true;
		        }
		        else if (szvalue.equalsIgnoreCase("false"))
		        {
		           bcontoDEF =  false;
		        }
		        else
		        {
                               szError += "Warning: "+szvalue +" is an unrecognized "+
				 "value for Include_Cellular_Process (expecting true or false)\n";
		        }
                     }
                     else if (sztype.equalsIgnoreCase("Permutation_Test_Should_Permute_Time_Point_0"))
 	             {
		        if (szvalue.equalsIgnoreCase("true"))
		        {
		           ballpermuteDEF = true;
		        }
		        else if (szvalue.equalsIgnoreCase("false"))
		        {
		           ballpermuteDEF =  false;
		        }
		        else
		        {
                            szError += "Warning: "+szvalue +" is an unrecognized "+
		       	       "value for Permutation_Test_Should_Permute_Time_Point_0 (expecting true or false)\n";
			}
		     }
                     else if (sztype.equalsIgnoreCase("Only_include_annotations_with_these_evidence_codes"))
	             {
		        szevidenceDEF = szvalue;
                     }
                     else if (sztype.equalsIgnoreCase("Only_include_annotations_with_these_taxon_IDs"))
	             {
	                sztaxonDEF = szvalue;
                     }
   	             else if (sztype.equalsIgnoreCase("Cross_Reference_Source"))
	             {
		        szxrefDEF = szvalue;
	                int numitems = GoAnnotations.defaultxrefs.length;
                        try
                        {
	                   nxrefDEF = Integer.parseInt(szxrefDEF);
	                   if ((nxrefDEF < 0)|| (nxrefDEF >= numitems))
	                   {
	                      nxrefDEF = 0;
	                   }
 	                }
                        catch(NumberFormatException ex)
                        {
	                   boolean bfound = false;
	                   int nsource = 0;
	                   while ((nsource < numitems)&&(!bfound))
	                   {
	                      if (((String) GoAnnotations.defaultxrefs[nsource]).equalsIgnoreCase(szxrefDEF))
	                      {
                                 bfound = true;
                                 nxrefDEF = nsource;
	                      }
  	                      else
	                      {
        	                 nsource++;
	                      }
	                   }   

		  	   if (!bfound)
		           {
		              szError += "Warning: "+szvalue +" is an unrecognized "+
				     "type for a Cross_Reference_Source\n";
		           }
	                }
	             }
	      	     else if (sztype.equalsIgnoreCase("Gene_Location_Source"))
		     {
	                szlocationDEF = szvalue;
	                int numitems = chromorganisms.length;
                        try
	                {
	                   nlocationDEF = Integer.parseInt(szlocationDEF);
	                   if ((nlocationDEF < 0)|| (nlocationDEF >= numitems))
	                   {
	                      nlocationDEF = 0;
	                   }
                        }
                        catch(NumberFormatException ex)
                        {
	                   boolean bfound = false;
                           int nsource = 0;
	                   while ((nsource < numitems)&&(!bfound))
	                   {
	                      if (((String) chromorganisms[nsource]).equalsIgnoreCase(szlocationDEF))
	                      {
                                 bfound = true;
                                 nlocationDEF = nsource;
                              }
  	                         else
	                         {
		                    nsource++;
	                         }
	                     }   

		  	    if (!bfound)
		            {
		                szError += "Warning: "+szvalue +" is an unrecognized "+
		       	       "type for a Location_Source\n";
		            }
	                }
		    }
    	            else if (sztype.equalsIgnoreCase("Cross_Reference_File"))
		    {
	               szCrossRefFileDEF = szvalue;
		    }
		    else if (sztype.equalsIgnoreCase("Gene_Location_File"))
		    {
		       szChromFileDEF = szvalue;
	            }
                    else if (sztype.equalsIgnoreCase("Data_File"))
	            {
                       szDataFileDEF = szvalue;
	            }
                    else if (sztype.equalsIgnoreCase("Category_ID_File"))
	            {
		       szcategoryIDDEF = szvalue;
	            }
                    else if (sztype.equalsIgnoreCase("Gene_Annotation_File"))
	            {
	               szGeneAnnotationFileDEF = szvalue;
	            }
  		    else if ((sztype.equalsIgnoreCase("Spot_IDs_included_in_the_the_data_file"))||
			        (sztype.equalsIgnoreCase("Spot_IDs_included_in_the_data_file"))||
			        (sztype.equalsIgnoreCase("Spot_IDs_in_the_data_file")))

	            {
	               bspotcheckDEF = (szvalue.equalsIgnoreCase("true"));
		    }
                    else if (sztype.equalsIgnoreCase("Comparison_Data_File"))
	  	    {
                       szCompareDEF = szvalue;
	            }
  		    else if ((sztype.equalsIgnoreCase("Comparison_Repeat_Data_Files(comma delimited list)"))||
	        	     (sztype.equalsIgnoreCase("Comparison_Repeat_Data_Files")))
      	            {
                       vRepeatCompareDEF = new Vector();
                       StringTokenizer stRepeatList = new StringTokenizer(szvalue, ",");
                       while (stRepeatList.hasMoreTokens())
	               {
		          vRepeatCompareDEF.add(stRepeatList.nextToken());
		       }
	            }
  		    else if ((sztype.equalsIgnoreCase("Comparison_Repeat_Data_is_from"))||
    (sztype.equalsIgnoreCase("Comparison_Repeat_Data_is_from[Different time periods,The same time period]")))
	            {
		       if (szvalue.equalsIgnoreCase("Different time periods"))
	               {
	                  bcomparealltimeDEF = true;
	               }
	  	       else if (szvalue.equalsIgnoreCase("The same time period"))
		       {
		          bcomparealltimeDEF = false;
	               }
  	               else if (!szvalue.equals("")) 
	               {
	                  szError += "WARNING: '"+szvalue+"' is an invalid value for "+
                                               "Comparison_Repeat_Data_is_from it must be either "+
                                              "'Different time periods' or 'The same time period'\n";
	               }
		    }
	            else if (sztype.equalsIgnoreCase("Comparison_Minimum_Number_of_genes_in_intersection"))
	            {
                       ntempval = Integer.parseInt(szvalue);
		       if (ntempval <= 0)
		       {
		          szError +="WARNING: '"+szvalue+
                               "' is an invalid value for Comparison_Minimum_Number_of_genes_in_intersection "+
                               "must be >=1\n";
		       }
		       else
		       {
		          nCompareMinGenesDEF = ntempval;
       		       }
                    }
  	            else if (sztype.equalsIgnoreCase("Comparison_Maximum_Uncorrected_Intersection_pvalue"))
	            {
                       dtempval = Double.parseDouble(szvalue);
		       if ((dtempval < 0)||(dtempval > 1))
		       {
                          szError +="WARNING: '"+szvalue+
                              "' is an invalid value for Comparison_Maximum_Uncorrected_Intersection_pvalue "+
                              "must be in [0,1]\n";
		       }
		       else
		       {
		          dCompareMinpvalDEF = dtempval;
                       }
	            }
	            else if (sztype.equalsIgnoreCase("Maximum_Number_of_Model_Profiles"))
	            {
                       ntempval = Integer.parseInt(szvalue);
		       if (ntempval < 0)
		       {
	                  szError +="WARNING: '"+szvalue+
                               "' is an invalid value for Maximum_Number_of_Model_Profiles "+
                              "must be >=0\n";
		       }
		       else
		       {
		          nMaxProfilesDEF = ntempval;
		       }
	            }
		    else if ((sztype.equalsIgnoreCase("Clustering_Method[STEM Clustering Method,K-means]"))||
			         (sztype.equalsIgnoreCase("Clustering_Method")))
		    {
		       if ((szvalue.equalsIgnoreCase("STEM Clustering Method"))||(szvalue.equalsIgnoreCase("STEM")))
	               {
                          nCLUSTERINGMETHODDEF = 0;
	               }
                       else if ((szvalue.equalsIgnoreCase("K-means"))||(szvalue.equalsIgnoreCase("Kmeans")))
		       {
                          nCLUSTERINGMETHODDEF = 1;
	               }
                       else if (!szvalue.equals(""))
		       {
		          szError += "WARNING: '"+szvalue+"' is an invalid value for "+
                                               "Clustering_Method it must be either 'STEM Clustering Method' "+
                                              "or 'K-means'\n";
		       }
	            }
	            else if ((sztype.equalsIgnoreCase("Number_of_Clusters"))||
			         (sztype.equalsIgnoreCase("Number_of_Clusters_K")))
		    {
		       ntempval = Integer.parseInt(szvalue);
		       if (ntempval <= 0)
		       {
                          szError +="WARNING: '"+szvalue+
                             "' is an invalid value for Number_of_Clusters_K must be >=1\n";
		       }
		       else
		       {
		          nKDEF = ntempval;
	       	       }
	            }
	            else if (sztype.equalsIgnoreCase("Number_of_Random_Starts"))
	            {
		       ntempval = Integer.parseInt(szvalue);
		       if (ntempval <= 0)
		       {
		          szError +="WARNING: '"+szvalue+
                            "' is an invalid value for Number_of_Random_Starts must be >=1\n";
		       }
		       else
		       {
		          nREPDEF = ntempval;
	      	       }
	            }
                    else if (sztype.equalsIgnoreCase("Maximum_Unit_Change_in_Model_Profiles_between_Time_Points"))
	            {
                       ntempval = Integer.parseInt(szvalue);
		       if (ntempval <= 0)
		       {
		          szError +="WARNING: '"+szvalue+
                              "' is an invalid value for Maximum_Unit_Change_in_Model_Profiles_"+
                              "between_Time_Points must be >=1\n";
		       }
		       else
		       {
		           nMaxUnitDEF = ntempval;
		       }
	            }
	 	    else if ((sztype.equalsIgnoreCase("Repeat_Data_Files(comma delimited list)"))||
	                       (sztype.equalsIgnoreCase("Repeat_Data_Files")))
		    {
                        vRepeatFilesDEF = new Vector();
                       StringTokenizer stRepeatList = new StringTokenizer(szvalue, ",");
                       while (stRepeatList.hasMoreTokens())
	               {
	                  vRepeatFilesDEF.add(stRepeatList.nextToken());
	               }
	            }
        	    else if ((sztype.equalsIgnoreCase("Repeat_Data_is_from"))||
                            (sztype.equalsIgnoreCase("Repeat_Data_is_from[Different time periods,The same time period]")))
	            {
	               if (szvalue.equalsIgnoreCase("Different time periods"))
	               {
	                  balltimeDEF = true;
	               }
  	               else if (szvalue.equalsIgnoreCase("The same time period"))
	               {
	                  balltimeDEF = false;
		       }
  		       else if (!szvalue.equals(""))
                       {
		           szError += "WARNING: '"+szvalue+"' is an invalid value for "+
                                               "Repeat_Data_is_from it must be either "+
                                              "'Different time periods' or 'The same time period'\n";
		       }
		    }
	            else if (sztype.equalsIgnoreCase("Maximum_Number_of_Missing_Values"))
	            {
                       ntempval = Integer.parseInt(szvalue);
		       if (ntempval < 0)
		       {
		          szError +="WARNING: '"+szvalue+
                             "' is an invalid value for Maximum_Number_of_Missing_Values must be >=0\n";
		       }
		       else
		       {
		          nMaxMissingDEF = ntempval;
		       }
                    }
  	            else if (sztype.equalsIgnoreCase("Minimum_Absolute_Log_Ratio_Expression"))
	            {
                       dtempval = Double.parseDouble(szvalue);
		       if (dtempval < -.05)
		       {
		          szError +="WARNING: '"+szvalue+
                          "' is an invalid value for Minimum_Absolute_Log_Ratio_Expression "+
                          "must be >= -.05\n";
		       }
	      	       else
	               {
		          dMinExpressionDEF = dtempval;
		       }
	            }
  	            else if (sztype.equalsIgnoreCase("Minimum_Correlation_between_Repeats"))
	            {
                        dtempval = Double.parseDouble(szvalue);
		        if ((dtempval < -1.1)||(dtempval > 1.1))
		        {
		           szError +="WARNING: '"+szvalue+
                              "' is an invalid value for Minimum_Correlation_between_Repeats "+
                              "must be in [-1.1,1.1]\n";
	     	        }
	               else
	               {
	                  dMinCorrelationRepeatsDEF  = dtempval;
	               }
		    }
 	            else if (sztype.equalsIgnoreCase("Pre-filtered_Gene_File"))
	            {
                       szPrefilteredDEF = szvalue;
	            }  
	            else if (sztype.equalsIgnoreCase("Maximum_Correlation"))
	            {
                       dtempval = Double.parseDouble(szvalue);
	   	       if ((dtempval < -1)||(dtempval > 1))
		       {
		          szError +="WARNING: '"+szvalue+
                                 "' is an invalid value for Maximum_Correlation "+
                                 "must be in [-1,1]\n";
	               }
	               else
                       {
	                  dMaxCorrelationModelDEF  = dtempval;
		       }
       	            }
	            else if (sztype.equalsIgnoreCase("Number_of_Permutations_per_Gene"))
	            {
         	       ntempval = Integer.parseInt(szvalue);
	      
		       if (ntempval < 0)
		       {
		          szError +="WARNING: '"+szvalue+
                                 "' is an invalid value for Number_of_Permutations_per_Gene must be >=0\n";
		       }
	               else
	               {
	                  nNumPermsGeneDEF = ntempval;
	               }
	            }
        	    else if (sztype.equalsIgnoreCase("Maximum_Number_of_Candidate_Model_Profiles"))
		    {
	               ntempval = Integer.parseInt(szvalue);
		       if (ntempval <= 0)
		       {
		          szError +="WARNING: '"+szvalue+
                                "' is an invalid value for Maximum_Number_of_Candidate_Model_Profiles must be >=1\n";
		       }
	       	       else
	               {
	                  nMaxCandidateModelDEF = ntempval;
		       }
	            }
  	            else if (sztype.equalsIgnoreCase("Significance_Level"))
	            {
                       dtempval = Double.parseDouble(szvalue);
		       if (dtempval < 0)
		       {
		          szError +="WARNING: '"+szvalue+
                                         "' is an invalid value for Significance_Level "+
                                         "must be >=0\n";
		       }
		       else
		       {
		          dSignificanceLevelDEF  = dtempval;
	       	       }
	            }
  	           else if (sztype.equalsIgnoreCase(
                            "Multiple_hypothesis_correction_method_enrichment[Bonferroni,Randomization]"))
	           {
                       if (szvalue.equalsIgnoreCase("Bonferroni"))
		       {
                          brandomgoDEF = false;
		       }
                       else if (szvalue.equalsIgnoreCase("Randomization"))
	               {
                           brandomgoDEF = true;
	               }
                       else if (!szvalue.equals(""))
		       {
		          szError += "WARNING: '"+szvalue+"' is an invalid value for "+
                                  "Multiple_hypothesis_correction_method_enrichment it must be either 'Bonferroni'"+
                                  "or 'Randomization'.\n";
		       }
	            }
  		    else if (sztype.equalsIgnoreCase("Correction_Method[Bonferroni,False Discovery Rate,None]"))
	            {
                       if (szvalue.equalsIgnoreCase("Bonferroni"))
		       {
                          nfdrDEF = 2;
		       }
                       else if (szvalue.equalsIgnoreCase("False Discovery Rate"))
		       {
                          nfdrDEF = 1;
	               }
                       else if (szvalue.equalsIgnoreCase("None"))
	               {
                          nfdrDEF = 0;
		       }
                       else if (!szvalue.equals(""))
	               {
		          szError += "WARNING: '"+szvalue+"' is an invalid value for "+
                                               "Correction_Method it must be either 'Bonferroni',"+
                                              "'False Discovery Rate', or 'None'\n";
		       }
	            }
	           else if (sztype.equalsIgnoreCase("Clustering_Minimum_Correlation"))
	           {
                      dtempval = Double.parseDouble(szvalue);
		      if ((dtempval < -1)||(dtempval > 1))
		      {
		         szError +="WARNING: '"+szvalue+
                                  "' is an invalid value for Clustering_Minimum_Correlation "+
                                  "must be in [-1,1]\n";
		      }
		      else
		      {
		         dMinimumCorrelationClusteringDEF  = dtempval;
		      }
	           }
	           else if (sztype.equalsIgnoreCase("Clustering_Minimum_Correlation_Percentile"))
	           {
                      dtempval = Double.parseDouble(szvalue);
		      if ((dtempval < 0)||(dtempval > 1))
		      {
		         szError +="WARNING: '"+szvalue+
                            "' is an invalid value for Clustering_Minimum_Correlation_Percentile "+
                           "must be in [0,1]\n";
		      }
		      else
		      {
		         dMinimumPercentileClusteringDEF  = dtempval;
		      }
	           }
  	           else if (sztype.equalsIgnoreCase("Number_of_samples_for_randomized_multiple_hypothesis_correction"))
	           {
                      ntempval = Integer.parseInt(szvalue);
		      if (ntempval <= 0)
		      {
	      	         szError +="WARNING: '"+szvalue+
                         "' is an invalid value for Number_of_samples_for_randomized_multiple_hypothesis_correction "+
                         "must be >=1\n";
		      }
		      else
		      {
		         nSamplesMultipleDEF = ntempval;
	              }
	           }
                   else if (sztype.equalsIgnoreCase("GO_Minimum_number_of_genes"))
	           {
                      ntempval = Integer.parseInt(szvalue);
		      if (ntempval <= 0)
		      {
		         szError +="WARNING: '"+szvalue+
                         "' is an invalid value for GO_Minimum_number_of_genes must be >=1\n";
		      }
		      else
		      {
		         nMinGoGenesDEF = ntempval;
	      	      }
    	           }
                   else if (sztype.equalsIgnoreCase("Minimum_GO_level"))
	           {
                      ntempval = Integer.parseInt(szvalue);
		      if (ntempval <= 0)
		      {
	      	          szError +="WARNING: '"+szvalue+
                          "' is an invalid value for Minimum_GO_Level must be >=1\n";
		      }
		      else
		      {
		         nMinGOLevelDEF = ntempval;
	              }
      	           }
                   else if (sztype.charAt(0)!='#')
      	           {
                      szError+= "WARNING: '"+sztype+"' is an unrecognized variable.\n";
	           }
		}
	     }
          }
       }
       br.close();

       if (!szError.equals(""))
       {
          throw new IllegalArgumentException(szError);
       }
    }

    ///////////////////////////////////////////////////////////////////////////////////
    /**
     * Makes sure repeat data sets agree with each other on the number of rows, columns,
     * and the name of the genes
     */
    public static void errorcheck(STEM_DataSet theDataSet1, STEM_DataSet theOtherSet)
    {
                   
       if (theDataSet1.numcols != theOtherSet.numcols)
       {
          throw new IllegalArgumentException(
                                            "Repeat data set must have same "+
                                            "number of columns as original, expecting "+
                                            theDataSet1.numcols+" found "+
                                            theOtherSet.numcols+" in the repeat");
       }   
       else if (theDataSet1.numrows != theOtherSet.numrows)
       {
	  throw new IllegalArgumentException(
                                            "Repeat data set must have same "+
                                            "number of spots as the original, expecting "+
                                            theDataSet1.numrows+" found "+
                                            theOtherSet.numrows+" in the repeat");
       }     
       else 
       {
          for (int nrow = 0; nrow < theDataSet1.numrows; nrow++)
	  {
	     if (!theDataSet1.genenames[nrow].equals(theOtherSet.genenames[nrow]))
	     {
	        throw new IllegalArgumentException("In row "+nrow+" of the repeat set "+
                                "expecting gene symbol "+theDataSet1.genenames[nrow] +
                                " found "+theOtherSet.genenames[nrow]);
	     }
	     else if (!theDataSet1.probenames[nrow].equals(theOtherSet.probenames[nrow]))
	     {
	        throw new IllegalArgumentException("In row "+nrow+" of the repeat set "+
                                "expecting gene symbol "+theDataSet1.probenames[nrow] +
                                " found "+theOtherSet.probenames[nrow]);
	     }                        
	  }
       }  
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Makes sure repeat data sets agree with each other on the number of columns
     * and the name of the genes 
     */
    public static void errorcheck(String[] origgenes, String[] repeatgenes, int norigcols, int nrepeatcols)
    {
                   
       if (norigcols != nrepeatcols)
       {
          throw new IllegalArgumentException(
                                            "Repeat data set must have same "+
                                            "number of columns as original, expecting "+
                                            norigcols+" found "+
                                            nrepeatcols+" in the repeat");
       }   
       else if (origgenes.length != repeatgenes.length)
       {
	  throw new IllegalArgumentException(
                                            "Repeat data set must have same "+
                                            "number of spots as the original, expecting "+
                                            origgenes.length+" found "+
                                            repeatgenes.length+" in the repeat");
       }     
       else 
       {
          for (int nrow = 0; nrow < origgenes.length; nrow++)
	  {
	     if (!origgenes[nrow].equals(repeatgenes[nrow]))
	     {
	        throw new IllegalArgumentException("In row "+nrow+" of the repeat set "+
                                "expecting gene symbol "+origgenes[nrow] +
                                " found "+repeatgenes[nrow]);
	     }
	     else if (!origgenes[nrow].equals(repeatgenes[nrow]))
	     {
	        throw new IllegalArgumentException("In row "+nrow+" of the repeat set "+
                                "expecting gene symbol "+origgenes[nrow] +
                                " found "+repeatgenes[nrow]);
	     }                        
	  }
       }  
    }

    //////////////////////////////////////////////////////////////////////////////////////
    /**
     * Record containing both a filtered data set and the names of the genes from the original data set
     */
    static class DataSetOrigRec
    {
	STEM_DataSet theDataSet;
	String[] origbasegenes;
	DataSetOrigRec(STEM_DataSet theDataSet,String[] origbasegenes)
	{
	    this.theDataSet = theDataSet;
	    this.origbasegenes = origbasegenes;
	}
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    /**
     * Calls buildsetwithOrig but only returns the data set, not the pre-filtered gene list
     */
    synchronized public  static STEM_DataSet buildset(
		      String szchromval,
		      String szxrefval,
                      String szexp1val,String szgoval, 
                      String szgocategoryval, int nmaxmissing,
                      double dexpressedval,double dmincorrelation,
                      double dlbcorrelationclust, double dalphaval,
                      double dpercentile, int nmaxchange,
                      int nnumberprofiles, double dcorrmodel,
                      int nsamplesgene, int nsamplespval, 
                      long nsamplesmodel,int nmingo, 
                      int nfdr,int nmingolevel,
                      String szextraval, boolean balltime, 
		      Vector repeatnames, boolean btakelog,
                      boolean bspotincluded,boolean badd0, String szcategoryIDval,
                      String szevidenceval,String sztaxonval,
                      boolean bpontoval,boolean bcontoval,
                      boolean bfontoval,boolean brandomgoval,boolean bkmeans,
                      boolean bmaxminval, boolean ballpermuteval,
                      String szorganismsourceval,String szxrefsourceval,
		      String szchromsourceval) throws Exception
    {

            DataSetOrigRec theRec = buildsetwithOrig(
			 szchromval,szxrefval,szexp1val,szgoval, 
                         szgocategoryval,  nmaxmissing,dexpressedval, dmincorrelation,
                         dlbcorrelationclust, dalphaval,dpercentile, nmaxchange,
                         nnumberprofiles, dcorrmodel,nsamplesgene, nsamplespval, 
                         nsamplesmodel, nmingo, nfdr, nmingolevel,
                         szextraval, balltime, repeatnames, btakelog,
                         bspotincluded, badd0, szcategoryIDval,szevidenceval, sztaxonval,bpontoval, bcontoval,
                         bfontoval, brandomgoval, bkmeans,bmaxminval, ballpermuteval,
                         szorganismsourceval, szxrefsourceval,
			 szchromsourceval);
	     return theRec.theDataSet;

    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * A driver method calling procedures to pre-process the data, and computer profile significance
     */
    synchronized public static DataSetOrigRec buildsetwithOrig(
		      String szchromval,
		      String szxrefval,
                      String szexp1val,String szgoval, 
                      String szgocategoryval, int nmaxmissing,
                      double dexpressedval,double dmincorrelation,
                      double dlbcorrelationclust, double dalphaval,
                      double dpercentile, int nmaxchange,
                      int nnumberprofiles, double dcorrmodel,
                      int nsamplesgene, int nsamplespval, 
                      long nsamplesmodel,int nmingo, 
                      int nfdr,int nmingolevel,
                      String szextraval, boolean balltime, 
		      Vector repeatnames, boolean btakelog,
                      boolean bspotincluded,boolean badd0, String szcategoryIDval,
                      String szevidenceval,String sztaxonval,
                      boolean bpontoval,boolean bcontoval,
                      boolean bfontoval,boolean brandomgoval,boolean bkmeans,
                      boolean bmaxminval, boolean ballpermuteval,
                      String szorganismsourceval,String szxrefsourceval,
                      String szchromsourceval) throws Exception
    {
	STEM_DataSet theDataSetsMerged = null;
      
	if (bkmeans)
	{
           if (balltime)
	   {
	       STEM_DataSet theDataSet1 = new STEM_DataSet(szexp1val, nmaxmissing, 
                                                dexpressedval, dmincorrelation, 
                                                nnumberprofiles, nmaxchange,
						 btakelog,bspotincluded,false,
                                                badd0,bmaxminval,balltime);

	      if (theDataSet1.numcols <=1)
	      {
	         //throw error if no prefiltered gene file
		  theDataSet1 = new STEM_DataSet(theDataSet1.filterDuplicates(),theDataSet1);
                  theDataSet1.tga = new STEM_GoAnnotations(szorganismsourceval,szxrefsourceval,szxrefval,szgoval,
                       szgocategoryval, theDataSet1.genenames, theDataSet1.probenames,nsamplespval, 
		       nmingo,nmingolevel,szextraval,szcategoryIDval,bspotincluded,
		       szevidenceval,sztaxonval,bpontoval,bcontoval,bfontoval,brandomgoval,szBatchGOoutput);

		  if (szBatchGOoutput != null)
		  {
		      theDataSet1.tga.clusterFileResults(szexp1val);
		  }

	         STEM_DataSet theDataSet1fm;
	         if (theDataSet1.numcols == 1)
	         {
		     theDataSet1fm = new STEM_DataSet(theDataSet1.filterMissing1point(),theDataSet1);
		    if (!badd0)
		    {
			theDataSet1fm = new STEM_DataSet(theDataSet1fm.filtergenesthreshold1point(),theDataSet1fm);
		    }
	         }
	         else
	         {
		    theDataSet1fm = theDataSet1;
	         }
	         theDataSet1fm.assignall0();
                 return new DataSetOrigRec(theDataSet1fm,theDataSet1.genenames);
	      }
	      else
	      {
	         String[] origgenes = theDataSet1.genenames;
	         theDataSet1 = new STEM_DataSet(theDataSet1.logratio2(),theDataSet1);
		 theDataSet1 = new STEM_DataSet(theDataSet1.averageAndFilterDuplicates(),theDataSet1);

	         //genevalues in log ratio before averaging stored 
                 //need for each gene duplicated 
                 //a mutlidimensional array of time series for each occurence

                 int numrepeats = repeatnames.size();       	

                 if (numrepeats > 0) 
	         {
                    STEM_DataSet[] repeatSets = new STEM_DataSet[numrepeats];
                    for (int nset = 0; nset < numrepeats; nset++)
	            {
		       String szfile = (String) repeatnames.get(nset);

	               STEM_DataSet theOtherSet = new STEM_DataSet(szfile, nmaxmissing, 
                                                 dexpressedval, dmincorrelation, 
					         nnumberprofiles, nmaxchange,
					      	 btakelog,bspotincluded,true,badd0,
							 bmaxminval,balltime);

		       errorcheck(origgenes, theOtherSet.genenames,theDataSet1.numcols,theOtherSet.numcols);   
		       //compute log ratio of each time series first then merge
                       //normalize the data  
	               theOtherSet =  new STEM_DataSet(theOtherSet.logratio2(),theOtherSet);
		       theOtherSet =  new STEM_DataSet(theOtherSet.averageAndFilterDuplicates(),theOtherSet);
		       //gene values in log ratio before averaging stored

                       repeatSets[nset]= theOtherSet; 
	            }
                    theDataSetsMerged =  new STEM_DataSet(theDataSet1.mergeDataSets(repeatSets),theDataSet1);
		    theDataSetsMerged =  new STEM_DataSet(theDataSetsMerged.filterdistprofiles(theDataSet1,repeatSets),theDataSetsMerged);
	         }
                 else
	         {
		    theDataSetsMerged = theDataSet1;	    
  	         }

  	         theDataSetsMerged = new STEM_DataSet(theDataSetsMerged.filterMissing(),theDataSetsMerged);
                 theDataSetsMerged =  new STEM_DataSet(theDataSetsMerged.filtergenesthreshold2(),theDataSetsMerged);
                 System.out.println("Number of selected genes is "+theDataSetsMerged.data.length);
		 theDataSetsMerged.kmeans();
       
                 theDataSetsMerged.tga = new STEM_GoAnnotations(szorganismsourceval,szxrefsourceval,szxrefval,szgoval,
                                  szgocategoryval,
                                  theDataSet1.genenames, theDataSet1.probenames,nsamplespval, 
		  		  nmingo,nmingolevel,szextraval,szcategoryIDval,bspotincluded,
					  szevidenceval,sztaxonval,bpontoval,bcontoval,bfontoval,brandomgoval,null);
                 theDataSetsMerged.tga.computeBestPvaluesClustersProfiles(theDataSetsMerged);
	         theDataSetsMerged.addExtraToFilter(theDataSetsMerged.tga);
		 
                 return new DataSetOrigRec(theDataSetsMerged, origgenes);
	      }
	   }
	   else
	   {
              //all the time series from same time period averaging then log ratio
	       STEM_DataSet theDataSet1 =  new STEM_DataSet(szexp1val, nmaxmissing, 
                                                dexpressedval, dmincorrelation, 
                                                nnumberprofiles, nmaxchange,
						  btakelog,bspotincluded,false,
                                                badd0, bmaxminval,balltime);

	      if (theDataSet1.numcols <=1)
	      {

		  theDataSet1 = new STEM_DataSet(theDataSet1.filterDuplicates(), theDataSet1);
		  theDataSet1.tga = new  STEM_GoAnnotations(szorganismsourceval,szxrefsourceval,
                                              szxrefval,szgoval,szgocategoryval,
                                              theDataSet1.genenames, theDataSet1.probenames,nsamplespval, 
					      nmingo,nmingolevel,szextraval,szcategoryIDval,bspotincluded,
					      szevidenceval,sztaxonval,bpontoval,
							    bcontoval,bfontoval,brandomgoval,szBatchGOoutput);

		  if (szBatchGOoutput != null)
		  {
		      theDataSet1.tga.clusterFileResults(szexp1val);
		  }

	         STEM_DataSet theDataSet1fm;
	         if (theDataSet1.numcols == 1)
	         {
		     theDataSet1fm = new STEM_DataSet(theDataSet1.filterMissing1point(),theDataSet1);
		    if (!badd0)
		    {
			theDataSet1fm =  new STEM_DataSet(theDataSet1fm.filtergenesthreshold1point(),theDataSet1fm);
		    }
	         }
	         else
	         {
		   theDataSet1fm = theDataSet1;
	         }
	         theDataSet1fm.assignall0();
                 return new DataSetOrigRec(theDataSet1fm,theDataSet1.genenames);
	      }
	      else
	      {
                 int numrepeats = repeatnames.size();       	
	   
                 if (numrepeats > 0)
	         {
		    STEM_DataSet[] repeatSets = new STEM_DataSet[numrepeats];
                    for (int nset = 0; nset < numrepeats; nset++)
	            {
		       String szfile = (String) repeatnames.get(nset);

	               STEM_DataSet theOtherSet = new STEM_DataSet(szfile, nmaxmissing, 
                                                dexpressedval, dmincorrelation, 
						nnumberprofiles, nmaxchange,
					        btakelog,bspotincluded,true,
						badd0, bmaxminval,balltime);

		       errorcheck(theDataSet1,theOtherSet);
    
                       repeatSets[nset]= theOtherSet; 
	             }
		    theDataSetsMerged = new STEM_DataSet(theDataSet1.mergeDataSets(repeatSets),theDataSet1);
	         }
                 else
	         {
		     theDataSetsMerged = new STEM_DataSet(theDataSet1,theDataSet1);	    
  	         }

  	         theDataSetsMerged =  new STEM_DataSet(theDataSetsMerged.logratio2(),theDataSetsMerged);
 	         theDataSetsMerged =  new STEM_DataSet(theDataSetsMerged.averageAndFilterDuplicates(),theDataSetsMerged);
	         //gene values before averaging stored
	         theDataSetsMerged =  new STEM_DataSet(theDataSetsMerged.filterMissing(),theDataSetsMerged);
                 theDataSetsMerged =  new STEM_DataSet(theDataSetsMerged.filtergenesthreshold2(),theDataSetsMerged);
                 System.out.println("Number of selected genes is "+theDataSetsMerged.data.length);
    
		 theDataSetsMerged.kmeans();
                 theDataSetsMerged.tga = new  STEM_GoAnnotations(szorganismsourceval,szxrefsourceval,
                                                szxrefval, szgoval, szgocategoryval,
                                                theDataSet1.genenames, theDataSet1.probenames,nsamplespval,
					      	 nmingo,nmingolevel,szextraval,szcategoryIDval,bspotincluded,
                                                szevidenceval,sztaxonval,bpontoval,
						 bcontoval,bfontoval,brandomgoval,null);
                 theDataSetsMerged.tga.computeBestPvaluesClustersProfiles(theDataSetsMerged);

	         theDataSetsMerged.addExtraToFilter(theDataSetsMerged.tga);
                 return new DataSetOrigRec(theDataSetsMerged,theDataSet1.genenames);
	      }
	   }
	}
	else
	{
           if (balltime)
	   {
              //all the time series represent full repeats
	       STEM_DataSet theDataSet1 = new STEM_DataSet(szexp1val, 
                                             nmaxmissing, dlbcorrelationclust, 
                                             dexpressedval, dmincorrelation, dalphaval,
                                             dpercentile, nmaxchange, nnumberprofiles, dcorrmodel,
					      nsamplesgene, nsamplesmodel,null,nfdr,
					      btakelog,bspotincluded,false,badd0, 
						 bmaxminval, ballpermuteval,balltime);

	      //System.out.println(theDataSet1.numcols);
	      if (theDataSet1.numcols <=1)
	      {
	         //throw error if no prefiltered gene file
		  theDataSet1 = new STEM_DataSet(theDataSet1.filterDuplicates(),theDataSet1);
		  theDataSet1.tga =  new  STEM_GoAnnotations(szorganismsourceval,szxrefsourceval,
		             	szxrefval,szgoval,szgocategoryval,
                               theDataSet1.genenames, theDataSet1.probenames,nsamplespval, 
		                 nmingo,nmingolevel,szextraval,szcategoryIDval,bspotincluded,
				    szevidenceval,sztaxonval,bpontoval,bcontoval,bfontoval,brandomgoval,szBatchGOoutput);

		  if (szBatchGOoutput != null)
		  {
		      theDataSet1.tga.clusterFileResults(szexp1val);
		  }

	         STEM_DataSet theDataSet1fm;
	         if (theDataSet1.numcols == 1)
	         {
		     theDataSet1fm = new STEM_DataSet(theDataSet1.filterMissing1point(),theDataSet1);
		    if (!badd0)
		    {
			theDataSet1fm = new STEM_DataSet(theDataSet1fm.filtergenesthreshold1point(),theDataSet1fm);
		    }
	         }
	         else
	         {
		    theDataSet1fm = theDataSet1;
	         }
	         theDataSet1fm.assignall0();
                 return new DataSetOrigRec(theDataSet1fm,theDataSet1.genenames);
	      }
	      else
	      {
	         String[] origgenes = theDataSet1.genenames;
	         theDataSet1 = new STEM_DataSet(theDataSet1.logratio2(),theDataSet1);
 	         theDataSet1 = new STEM_DataSet(theDataSet1.averageAndFilterDuplicates(),theDataSet1);

	         //genevalues in log ratio before averaging stored 
                 //need for each gene duplicated 
                 //a mutlidimensional array of time series for each occurence

                 int numrepeats = repeatnames.size();       	

                 if (numrepeats > 0) 
	         {
		    STEM_DataSet[] repeatSets = new STEM_DataSet[numrepeats];
                    for (int nset = 0; nset < numrepeats; nset++)
	            {
		       String szfile = (String) repeatnames.get(nset);
		       STEM_DataSet theOtherSet = new STEM_DataSet(szfile, 
                                             nmaxmissing,dlbcorrelationclust, 
                                             dexpressedval, dmincorrelation, dalphaval,
                                             dpercentile, nmaxchange, nnumberprofiles, 
			                     dcorrmodel, nsamplesgene,nsamplesmodel,
					     theDataSet1.modelprofiles,nfdr,btakelog,
					     bspotincluded,true,badd0, 
                                             bmaxminval, ballpermuteval,balltime);
		       errorcheck(origgenes, theOtherSet.genenames,theDataSet1.numcols,theOtherSet.numcols);   
		       //compute log ratio of each time series first then merge
                       //normalize the data  
	               theOtherSet = new STEM_DataSet(theOtherSet.logratio2(),theOtherSet);
  	               theOtherSet = new STEM_DataSet(theOtherSet.averageAndFilterDuplicates(),theOtherSet);
		       //gene values in log ratio before averaging stored

                       repeatSets[nset]= theOtherSet; 
	            }
                    theDataSetsMerged = new STEM_DataSet(theDataSet1.mergeDataSets(repeatSets),theDataSet1);//,true);
                    theDataSetsMerged = new STEM_DataSet(theDataSetsMerged.filterdistprofiles(theDataSet1,repeatSets),theDataSetsMerged);
	         }
                 else
	         {
		    theDataSetsMerged = theDataSet1;	    
  	         }

  	         theDataSetsMerged =  new STEM_DataSet(theDataSetsMerged.filterMissing(),theDataSetsMerged);
                 theDataSetsMerged =  new STEM_DataSet(theDataSetsMerged.filtergenesthreshold2(),theDataSetsMerged);
                 System.out.println("Number of selected genes is "+theDataSetsMerged.data.length);
                 theDataSetsMerged.findbestgroupassignments();
	         theDataSetsMerged.tallyassignments();       
                 theDataSetsMerged.computeaveragetally();
                 theDataSetsMerged.computePvaluesAssignments();
       
                 theDataSetsMerged.clusterprofiles(theDataSetsMerged.significantnum,
                             theDataSetsMerged.clustersofprofilesnum, true);
	    
                 theDataSetsMerged.tga = new  STEM_GoAnnotations(szorganismsourceval,szxrefsourceval,
                                  szxrefval,szgoval, szgocategoryval,
                                  theDataSet1.genenames, theDataSet1.probenames,nsamplespval, 
		  		  nmingo,nmingolevel,szextraval,szcategoryIDval,bspotincluded,
				  szevidenceval,sztaxonval,bpontoval,bcontoval,bfontoval,brandomgoval,null);
                 theDataSetsMerged.tga.computeBestPvaluesClustersProfiles(theDataSetsMerged);

	         theDataSetsMerged.addExtraToFilter(theDataSetsMerged.tga);
                 return new DataSetOrigRec(theDataSetsMerged,theDataSet1.genenames);
	      }
	  }
	  else
          {
             //all the time series from same time period averaging then log ratio
	      STEM_DataSet theDataSet1 = new STEM_DataSet(szexp1val,
                                             nmaxmissing, dlbcorrelationclust, 
                                             dexpressedval, dmincorrelation, dalphaval,
                                             dpercentile, nmaxchange, nnumberprofiles, dcorrmodel,
					      nsamplesgene, nsamplesmodel,null,nfdr,
					      btakelog,bspotincluded,false,badd0,
					      bmaxminval, ballpermuteval,balltime);
	     if (theDataSet1.numcols <=1)
	     {

		 theDataSet1 = new STEM_DataSet(theDataSet1.filterDuplicates(),theDataSet1);
		 theDataSet1.tga = new STEM_GoAnnotations(szorganismsourceval,szxrefsourceval,
                                              szxrefval,szgoval,szgocategoryval,
                                              theDataSet1.genenames, theDataSet1.probenames,nsamplespval, 
						   nmingo,nmingolevel,szextraval,szcategoryIDval,bspotincluded,
						   szevidenceval,sztaxonval,bpontoval,
							  bcontoval,bfontoval,brandomgoval,szBatchGOoutput);

	        if (szBatchGOoutput != null)
	        {
	           theDataSet1.tga.clusterFileResults(szexp1val);
	        }
	        STEM_DataSet theDataSet1fm;
	        if (theDataSet1.numcols == 1)
	        {
	            theDataSet1fm =  new STEM_DataSet(theDataSet1.filterMissing1point(),theDataSet1);
		    if (!badd0)
		    {
			theDataSet1fm =  new STEM_DataSet(theDataSet1fm.filtergenesthreshold1point(),theDataSet1fm);
		    }
	        }
	        else
	        {
		   theDataSet1fm = theDataSet1;
	        }
	        theDataSet1fm.assignall0();
                return new DataSetOrigRec(theDataSet1fm,theDataSet1.genenames);
	     }
	     else
	     {
                int numrepeats = repeatnames.size();       	
	   
                if (numrepeats > 0)
	        {
                   STEM_DataSet[] repeatSets = new STEM_DataSet[numrepeats];
                   for (int nset = 0; nset < numrepeats; nset++)
	           {
		      String szfile = (String) repeatnames.get(nset);
		      STEM_DataSet theOtherSet = new STEM_DataSet(szfile,
                                             nmaxmissing,dlbcorrelationclust,
                                             dexpressedval, dmincorrelation, dalphaval,
                                             dpercentile, nmaxchange, nnumberprofiles, 
			                     dcorrmodel, nsamplesgene,nsamplesmodel,
					     theDataSet1.modelprofiles,nfdr,
					     btakelog,bspotincluded,true,badd0,
					     bmaxminval, ballpermuteval,balltime);   

		      errorcheck(theDataSet1,theOtherSet);
    
                      repeatSets[nset]= theOtherSet; 
	           }
                   theDataSetsMerged =  new STEM_DataSet(theDataSet1.mergeDataSets(repeatSets),theDataSet1);//,false);
	        }
                else
	        {
		   theDataSetsMerged = theDataSet1;	    
  	        }

  	        theDataSetsMerged =  new STEM_DataSet(theDataSetsMerged.logratio2(),theDataSetsMerged);
 	        theDataSetsMerged =  new STEM_DataSet(theDataSetsMerged.averageAndFilterDuplicates(),theDataSetsMerged);
	        //gene values before averaging stored
	        theDataSetsMerged =  new STEM_DataSet(theDataSetsMerged.filterMissing(),theDataSetsMerged);
                theDataSetsMerged =  new STEM_DataSet(theDataSetsMerged.filtergenesthreshold2(),theDataSetsMerged);
                System.out.println("Number of selected genes is "+theDataSetsMerged.data.length);
                theDataSetsMerged.findbestgroupassignments();
	        theDataSetsMerged.tallyassignments();       
                theDataSetsMerged.computeaveragetally();
                theDataSetsMerged.computePvaluesAssignments();      
                theDataSetsMerged.clusterprofiles(theDataSetsMerged.significantnum,
                                    theDataSetsMerged.clustersofprofilesnum, true);

                theDataSetsMerged.tga = new STEM_GoAnnotations(szorganismsourceval,szxrefsourceval,
                                          szxrefval, szgoval,szgocategoryval,
                                                theDataSet1.genenames, theDataSet1.probenames,nsamplespval,
					      	 nmingo,nmingolevel,szextraval,szcategoryIDval,bspotincluded,
                                                szevidenceval,sztaxonval,bpontoval,
					       bcontoval,bfontoval,brandomgoval,null);
                theDataSetsMerged.tga.computeBestPvaluesClustersProfiles(theDataSetsMerged);
	     }
	     theDataSetsMerged.addExtraToFilter(theDataSetsMerged.tga);
             return new DataSetOrigRec(theDataSetsMerged,theDataSet1.genenames);
	  }
	}
    }

    ///////////////////////////////////////////////////////////////////////////
    /**
     * Prints a table for batch output about each profile or k-means cluster
     */
    public void printBatchOutputProfileTable(STEM_DataSet theDataSet) throws IOException
    {

        PrintWriter pw;

	if (theDataSet.bkmeans)
	{
           pw = new PrintWriter(new FileWriter(szBatchOutputDir+File.separator+
			       Util.stripLastExtension(szcurrentDefaultFile)+"_kmeansclustertable.txt"));
           pw.println("Cluster"+"\tCluster Mean\tNumber of Genes");
	   for (int nprofile = 0; nprofile < theDataSet.countassignments.length; nprofile++)
	   {
	      pw.print(nprofile+"\t");
	      pw.print(theDataSet.modelprofiles[nprofile][0]);
	      for (int ncol = 1; ncol < theDataSet.modelprofiles[nprofile].length; ncol++)
	      {
	         pw.print(","+theDataSet.modelprofiles[nprofile][ncol]);
	      }
              pw.println("\t"+theDataSet.countassignments[nprofile]);
	   }
	}
	else
	{
           pw = new PrintWriter(new FileWriter(szBatchOutputDir+File.separator+
			      Util.stripLastExtension(szcurrentDefaultFile)+"_profiletable.txt"));
           pw.print("Profile ID"+"\tProfile Model\tCluster (-1 non-significant)");	
           pw.println("\t# Genes Assigned\t# Gene Expected\tp-value");
	   int[] clusterassign = new int[theDataSet.countassignments.length];
	   for (int nel = 0; nel < clusterassign.length; nel++)
	   {
	      clusterassign[nel] = -1;
	   }

	   for (int ncluster = 0; ncluster < theDataSet.clustersofprofilesnum.size(); ncluster++)
	   {
	      ArrayList al = (ArrayList) theDataSet.clustersofprofilesnum.get(ncluster);
	      for (int nel = 0; nel < al.size(); nel++)
	      {
	         clusterassign[((STEM_DataSet.ProfileRec) al.get(nel)).nprofileindex] = ncluster;
	      }
	   }
	 
	   for (int nprofile = 0; nprofile < theDataSet.countassignments.length; nprofile++)
	   {
	      pw.print(nprofile+"\t");
	      pw.print(theDataSet.modelprofiles[nprofile][0]);
	      for (int ncol = 1; ncol < theDataSet.modelprofiles[nprofile].length; ncol++)
	      {
	         pw.print(","+theDataSet.modelprofiles[nprofile][ncol]);
	      }
              pw.println("\t"+clusterassign[nprofile]+"\t"+theDataSet.countassignments[nprofile]+"\t"+
        	         theDataSet.expectedassignments[nprofile]+"\t"+
                         Util.doubleToSz(theDataSet.pvaluesassignments[nprofile]));
	   }
	}
	pw.close();
    }

    //////////////////////////////////////////////////////////////////////////
    /**
     * Prints a table for batch output showing for each gene the profile to which it was assigned
     */
    public void printBatchOutputGeneTable(STEM_DataSet theDataSet) throws IOException
    {

        PrintWriter pw = new PrintWriter(new FileWriter(szBatchOutputDir+File.separator+
				 Util.stripLastExtension(szcurrentDefaultFile)+"_genetable.txt"));

        pw.print(theDataSet.szGeneHeader+"\t"+theDataSet.szProbeHeader+"\t");
	if (theDataSet.bkmeans)
	{
	   pw.print("Cluster");
	}
	else
	{
	   pw.print("Profile");
	}

        for (int ncolindex = 0; ncolindex < theDataSet.numcols; ncolindex++)
	{
	    pw.print("\t"+theDataSet.dsamplemins[ncolindex]);
	}
	pw.println();

       for (int nindex = 0; nindex < theDataSet.numrows; nindex++)
       {
          pw.print(theDataSet.genenames[nindex]+"\t"+theDataSet.probenames[nindex]+"\t");
          ArrayList bestAssignments = theDataSet.bestassignments[nindex];
          pw.print(""+((Integer) bestAssignments.get(0)).intValue());
          for (int njindex = 1; njindex < bestAssignments.size(); njindex++)
	  {
	     pw.print(";"+((Integer) bestAssignments.get(njindex)).intValue());
          }

          NumberFormat nf2 = NumberFormat.getInstance(Locale.ENGLISH);
          nf2.setMinimumFractionDigits(2);
          nf2.setMaximumFractionDigits(2);

          for (int ncol = 0; ncol < theDataSet.numcols-1; ncol++)
	  {
             if (theDataSet.pmavalues[nindex][ncol]==0)
	     {
	        pw.print("\t");
             }
             else
	     {
       	        pw.print("\t"+nf2.format(theDataSet.data[nindex][ncol]));
	     }
	  }
	 
	  pw.println("\t"+nf2.format(theDataSet.data[nindex][theDataSet.numcols-1]));     
       }
       pw.close();
    }

    /////////////////////////////////////////////////////////////////////////////
    /**
     * The method which drives the execution after the execute button is pressed
     * getting the input and buiding the data sets
     */
    public void clusterscript(
		      String szchromval,
		      String szxrefval,
                      String szexp1val,String szexp2val,String szgoval, 
                      String szgocategoryval, String szmaxmissingval,
                      String szexpressedval,String szfilterthresholdval,
                      String szlbval, String szalphaval,
                      String szpercentile, String szmaxchange,
                      String sznumberprofiles, String szcorrmodel,
                      String szsamplegene,String szsamplepval, 
                      String szsamplemodel,String szmingoval, 
                      int nfdr,String szmingolevelval,
                      String szextraval, boolean balltime, 
                      Vector repeatnames, boolean btakelog,
                      boolean bgetchromval, boolean bgetxref, boolean bgetgoann,
		      boolean bspotincluded, boolean badd0,
                      String szcategoryIDval,
                      String szevidenceval,String sztaxonval,
                      boolean bpontoval,boolean bcontoval,
                      boolean bfontoval,boolean brandomgoval,
                      boolean bmaxminval,boolean ballpermuteval) throws Exception
    {

	synchronized (lockpd)
	{
	    while (ntodownload > 0)
	    {
		lockpd.wait();
	    }
	}

	if (nexceptions == 0)
	{
	   if (szexp1val.trim().equals(""))
	   {
              throw new IllegalArgumentException("No input data file given!");
	   }
	   else if (!(new File(szexp1val)).exists())
	   {
              throw new IllegalArgumentException("The input data file '"+szexp1val+"' cannot be found."); 
	   }


           if (szcategoryIDval.trim().equals(""))
	   {
	      szcategoryIDval = "";
	   }
	   else if (!(new File(szcategoryIDval)).exists())
	   {
              throw new IllegalArgumentException("The category ID file '"+szcategoryIDval+"' cannot be found."); 
	   }

	   if (szchromval.trim().equals(""))
	   {
	       szchromval = "";
	   }
	   else if ((!bgetchromval)&&!(new File(szchromval)).exists())
	   {
            throw new IllegalArgumentException("The gene location file '"+szchromval+"' cannot be found."); 
	   }

           if (szxrefval.trim().equals(""))
	   {
	      szxrefval = "";
	   }
	   else if ((!bgetxref)&&!(new File(szxrefval)).exists())
	   {
              throw new IllegalArgumentException("The cross reference file '"+szxrefval+"' cannot be found."); 
	   }

           if (szgoval.trim().equals(""))
	   {
	      szgoval = "";
	   }
	   else if ((!bgetgoann)&&(!(new File(szgoval)).exists()))
	   {
              throw new IllegalArgumentException("The GO annotation file '"+szgoval+"' cannot be found."); 
	   }
         

           if (szextraval.trim().equals(""))
	   {
	      szextraval = "";
	   }
	   else if (!(new File(szextraval)).exists())
	   {
              throw new IllegalArgumentException("The pre-filtered gene list file '"+
                                                szextraval+"' cannot be found."); 
	   }

           if (szgocategoryval.trim().equals(""))
	   {
	      szgocategoryval = "";
	   }

	   STEM_DataSet theDataSetfmnel;
           int nmaxmissing;
           try
           {
              nmaxmissing = Integer.parseInt(szmaxmissingval);
              if (nmaxmissing < 0)
              {
                 throw new IllegalArgumentException("Maximum missing values must be positive");
              }
           }
           catch (NumberFormatException ex)
           {
              throw new IllegalArgumentException("Maximum missing values must be an integer");
           }

           for (int nrepeat = 0; nrepeat < repeatnames.size(); nrepeat++)
	   { 
	       if (!(new File((String) repeatnames.get(nrepeat))).exists())
	       {
		   throw new IllegalArgumentException("The repeat data file '"+repeatnames.get(nrepeat)+
                                                        "' cannot be found");

	       }
	   }

           double dmincorrelation = Double.parseDouble(szfilterthresholdval);
           if ((dmincorrelation < -1.1)||(dmincorrelation > 1.1))
           {
              throw new IllegalArgumentException(
                     "Correlation Lower Bound for Filtering must be in [-1.1,1.1]");
           }

           double dexpressedval = Double.parseDouble(szexpressedval);
           if (dexpressedval < -0.05)
           {
              throw new IllegalArgumentException("Expression Value for filter must be >= -0.05");
           }

           double dlbcorrelationclust = Double.parseDouble(szlbval);
           if ((dmincorrelation < -1.1)||(dmincorrelation > 1.1))
           {
              throw new IllegalArgumentException(
                      "Minimum Correlation for Clustering must be in [-1,1]");
           }

           double dalphaval = Double.parseDouble(szalphaval);
           if (dalphaval < 0)
           {
	      throw new IllegalArgumentException("Alpha value must non-negative");
           }

           double dpercentile = Double.parseDouble(szpercentile);
           if ((dpercentile < 0)||(dpercentile > 1))
           {
              throw new IllegalArgumentException(
                     "Minimum Correlation Percentile for Clustering must be in [0,1]");
           }

           int nmingo = Integer.parseInt(szmingoval);
           if (nmingo< 1)
           {
              throw new IllegalArgumentException("Minimum number of GO genes must be at least 1");
           }

           int nmingolevel = Integer.parseInt(szmingolevelval);
           if (nmingolevel< 1)
           {
              throw new IllegalArgumentException("Minimum GO level must be at least 1");
           }

           int  nmaxchange;
 
           try
           {
              nmaxchange  = Integer.parseInt(szmaxchange);
           }
           catch (NumberFormatException ex)
           {
              throw new IllegalArgumentException(
                   "Maximum unit change in model profiles must be an integer");
           }

           if (nmaxchange < 1)
           {
              throw new IllegalArgumentException("Maximum unit change must be >= 1"); 
           }
 
           int  nnumberprofiles;
   
           try
           {
              nnumberprofiles = Integer.parseInt(sznumberprofiles);
           }
           catch (NumberFormatException ex)
           {
              throw new IllegalArgumentException("Maximum number of profiles must be an integer");
           } 
 
           if (nnumberprofiles < 0)
           {
              throw new IllegalArgumentException("Maximum number of profiles must be >= 0"); 
           }

           double dcorrmodel = Double.parseDouble(szcorrmodel);
           if ((dcorrmodel < -1)||(dcorrmodel > 1))
           {
              throw new IllegalArgumentException(
                 "Maximum correlation between model profiles must be in [-1,1]");
           }

           int nsamplesgene;

           try
           {
              nsamplesgene = Integer.parseInt(szsamplegene);
           }
           catch (NumberFormatException ex)
           {
              throw new IllegalArgumentException("Number of permutations per gene must be an integer");
           }

           if (nsamplesgene < 0)
           {
              throw new IllegalArgumentException("Number of permutations per gene must be >=0"); 
           }

           int nsamplespval;
 
           try
           {
              nsamplespval = Integer.parseInt(szsamplepval);
           } 
           catch (NumberFormatException ex)
           {
              throw new IllegalArgumentException("Number of samples for p-value correction must be an integer");
           }

           if (nsamplespval < 1)
           {
              throw new IllegalArgumentException("Number of samples for p-value correction must be positive"); 
           }

           long nsamplesmodel;
           try
           {
              nsamplesmodel = Integer.parseInt(szsamplemodel);
           }
           catch (NumberFormatException ex)
           {
              throw new IllegalArgumentException("Maximum number of candidate model profiles must be an integer");
           }

           if (nsamplesmodel <  nnumberprofiles)
           {
              throw new IllegalArgumentException(
                      "Maximum number of candidate model profiles must be >= maximum number of profiles");
	   }

           if (nsamplesmodel < 1)
           {
              throw new IllegalArgumentException("Maximum number of candidate model profiles must be positive");
           }

	   if (!bbatchmode)
	   {
              executeDialognf = new JDialog(this, "Executing...");
	  
              final JDialog executeDialog = executeDialognf; 
	      final JFrame thisframe = this;
              javax.swing.SwingUtilities.invokeAndWait(new Runnable() 
              {
                 public void run() 
                 {
	            Container cp = executeDialog.getContentPane();
		    JPanel lp = new JPanel(new SpringLayout());
		    lp.setBackground(Color.white);
		    lp.add(new JLabel("Please wait while STEM analyzes the data."));

                    SpringUtilities.makeCompactGrid(lp,1,1,10,10,10,10);	         
                    cp.add(lp);
		    executeDialog.pack();
                    executeDialog.setBackground(Color.white);
                    cp.setBackground(Color.white); 
                    executeDialog.setLocation(thisframe.getX()+250,thisframe.getY()+200);    
                    executeDialog.setDefaultCloseOperation(DISPOSE_ON_CLOSE);
                    executeDialog.setVisible(true);
		 }
              });
	   }

           final DataSetOrigRec theDataSetOrigRec  = ST.buildsetwithOrig(
		    szchromval,szxrefval,
                    szexp1val, szgoval, szgocategoryval, nmaxmissing,
                    dexpressedval, dmincorrelation, dlbcorrelationclust, dalphaval,
                    dpercentile, nmaxchange, nnumberprofiles, dcorrmodel,
                    nsamplesgene, nsamplespval,nsamplesmodel, nmingo, 
                    nfdr, nmingolevel, szextraval, balltime, 
                    repeatnames,btakelog,bspotincluded,badd0,szcategoryIDval,
                    szevidenceval,sztaxonval,bpontoval,bcontoval,bfontoval,brandomgoval,
                    (nclusteringmethodcb>0),bmaxminval, ballpermuteval,
                    szorganismsourceval, szxrefsourceval,szchromsourceval);

	   final STEM_DataSet thefDataSetfmnel = theDataSetOrigRec.theDataSet;
	   final String[] origbasegenes = theDataSetOrigRec.origbasegenes;


	   if (bbatchmode)
	   {
	       File f = new File(szBatchOutputDir);
	       if (!f.exists())
	       {
	          f.mkdir();
	       }

	       printBatchOutputGeneTable(thefDataSetfmnel);
	       printBatchOutputProfileTable(thefDataSetfmnel);
	   }
	   else
	   {
	      GenomeFileParser gfp=null;

              try
	      {
	         gfp = new GenomeFileParser(szchromval,szchromsourceval,origbasegenes,
		       thefDataSetfmnel.tga.extragenes,thefDataSetfmnel.tga.brandomgoval,
                       thefDataSetfmnel.tga.nsamplespval);
	      }
	      catch (Exception ex)
	      {
	         //not going to bail because locations not working, pretend they were not provided
	         gfp = new GenomeFileParser();
	         gfp.szchromval = "";
	         ex.printStackTrace(System.out);
	      }
	      
	      final GenomeFileParser fgfp = gfp;

	      if (thefDataSetfmnel.numcols >= 2)
	      {
	      
                 javax.swing.SwingUtilities.invokeAndWait(new Runnable() 
                 {
                    public void run() 
                    {
		       final MAINGUI2 f = new MAINGUI2(thefDataSetfmnel);
		       f.chromViewInit(fgfp);
		       edu.umd.cs.piccolo.PCanvas.CURRENT_ZCANVAS = null;
                       f.setLocation(15,40);
                       f.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		       f.setVisible(true);
                       f.addWindowListener(new WindowAdapter() {
                       public void windowClosing(WindowEvent we) {
		          f.closeSortWindows();
		       }
                       public void windowClosed(WindowEvent we) {
			  StatUtil.htbinom = new Hashtable();
                          Thread t = new Thread (new Runnable() { 
                                      public void run() { 
					   System.gc();     
                                      } 
                                  } 
                           ); 
                           t.start(); 
		        }
  	             }); 
                   }
                });
	      }
  	      else
	      {
                 javax.swing.SwingUtilities.invokeAndWait(new Runnable() 
                 {
                    public void run() 
                    {
		       ChromFrame cf;
		       if (fgfp != null && fgfp.parsedOK()) 
		       {
		         cf = new ChromFrame(thefDataSetfmnel.szGeneHeader,fgfp,1);	
			 cf.setVisible(true);
			 cf.drawGenes(thefDataSetfmnel.genenames);    
		       } 
		       else
		       {
			   cf = null;
		       }

		       if  ((thefDataSetfmnel.tga.szGoFile != null)&&
			   (!thefDataSetfmnel.tga.szGoFile.equals("")))
		      {
		         String szTitle ="GO Results";
                         JFrame frame = new JFrame(szTitle);
                         frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		         frame.setLocation(25,100);
	 	         GOTable newContentPane = 
                           new GOTable(frame,thefDataSetfmnel, 0,null,false,null,szTitle,true,cf);
                         newContentPane.setOpaque(true); //content panes must be opaque
                         frame.setContentPane(newContentPane);

                         frame.pack();
                         frame.setVisible(true);	
		         frame.toFront();
		      }
		    }
	         });
	      }
	   
              if (executeDialognf != null)
              {
	         executeDialognf.setVisible(false);
		 executeDialognf.dispose();
	      }

	      long e1 = System.currentTimeMillis();
	      System.out.println("Time: "+(e1-s1)+"ms");  
	   }
	}   
    }


    /**
     * Responds to changes in the desired clustering method on the interface
     */
    public void handleclusteringmethod()
    {
	nclusteringmethodcb = clusteringmethodcb.getSelectedIndex();

	if ((nclusteringmethodcb >= 1)&&(noldcluteringmethodcb == 0))
	{
            try
	    {
	       thespinnerChange.commitEdit();
	       thespinnerProfile.commitEdit();
	    }
            catch (ParseException ex)
	    {
		System.out.println("Warning: could not parse number of profiles");
	    }
	    SpinnerNumberModel snm = (SpinnerNumberModel) thespinnerProfile.getModel();
	    nMaxProfiles = ((Integer) snm.getValue()).intValue();

	    snm.setValue(new Integer(nK));
	    snm.setMinimum(new Integer(1));
	    snm.setStepSize(new Integer(1));

	    SpinnerNumberModel snm2 = (SpinnerNumberModel) thespinnerChange.getModel();
	    nMaxChangeProfiles = ((Integer) snm2.getValue()).intValue();
	    snm2.setValue(new Integer(nreps));
	    maxchangeLabel.setText(szrep);
	    numberprofilesLabel.setText(szk);
	}
	else if ((nclusteringmethodcb == 0) && (noldcluteringmethodcb >= 1))
	{
	    //switching to STEM clustering method

            try
	    {
	       thespinnerChange.commitEdit();
	       thespinnerProfile.commitEdit();
	    }
            catch (ParseException ex)
	    {
		System.out.println("Warning: could not parse number of clusters");
	    }

	    SpinnerNumberModel snm = (SpinnerNumberModel) thespinnerProfile.getModel();
	    nK = ((Integer) snm.getValue()).intValue();
	    snm.setValue(new Integer(nMaxProfiles));
	    snm.setMinimum(new Integer(0));

	    SpinnerNumberModel snm2 = (SpinnerNumberModel) thespinnerChange.getModel();
	    nreps = ((Integer) snm2.getValue()).intValue();
	    snm2.setValue(new Integer(nMaxChangeProfiles));
	    snm.setStepSize(new Integer(5));

	    numberprofilesLabel.setText(sznumberprofiles);
	    maxchangeLabel.setText(szmaxchange);
	}
	noldcluteringmethodcb = nclusteringmethodcb;
    }

    ////////////////////////////////////////////////////////////////////////////////////
    /**
     * Responds to changes in the gene annotation cross reference field on the main interface
     */
    public void handlenxrefval()
    {
       if (xrefField.isEditable())
       {
          szuserxref = xrefField.getText();
       }

       nxrefcb = xrefcb.getSelectedIndex();
       szxrefsourceval = GoAnnotations.defaultxrefs[nxrefcb];
       if (nxrefcb>=2)
       {
          xrefField.setText(GoAnnotations.xreffile[nxrefcb]);
          xrefField.setEditable(false);
          xrefButton.setEnabled(false);
       }
       else if ((nxrefcb == 0) && (ndb != 1))
       {
          xrefField.setText(szuserxref);
          xrefField.setEditable(true);
          xrefButton.setEnabled(true);
       }
       else 
       {
          xrefField.setText("");
          xrefField.setEditable(false);
          xrefButton.setEnabled(false);
       }

       if ((nxrefcb>=2)&&(!GoAnnotations.xreffile[nxrefcb].equals("")))
       {
          File xrefFile = new File(GoAnnotations.xreffile[nxrefcb]);
	  if (xrefFile.exists())
	  {
	     xrefcheck.setSelected(false);
	     xrefcheck.setEnabled(true);
	  }
	  else
	  {
	     xrefcheck.setSelected(true);
	     xrefcheck.setEnabled(false);
	  }
       }
       else
       {
          xrefcheck.setSelected(false);
          xrefcheck.setEnabled(false);
       }
    }


    /////////////////////////////////////////////////////////////////////////
    /**
     * Responds to changes in the gene location field on the input interface
     */
    public void handlechromval()
    {
       if (chromField.isEditable())
       {
          szuserchrom = chromField.getText();
       }

       nchromcb = chromcb.getSelectedIndex();
       szchromsourceval = chromorganisms[nchromcb];       

       if (nchromcb>=2)
       {
          chromField.setText(chromfile[nchromcb]);
          chromField.setEditable(false);
          chromButton.setEnabled(false);
       }
       else if (nchromcb == 0)
       {
          chromField.setText(szuserchrom);
          chromField.setEditable(true);
          chromButton.setEnabled(true);
       }
       else 
       {
          chromField.setText("");
          chromField.setEditable(false);
          chromButton.setEnabled(false);
       }

       if ((nchromcb>=2)&&(!chromfile[nchromcb].equals("")))
       {
          File chromFile = new File(chromfile[nchromcb]);
	  if (chromFile.exists())
	  {
	     chromcheck.setSelected(false);
	     chromcheck.setEnabled(true);
	  }
	  else
	  {
	     chromcheck.setSelected(true);
	     chromcheck.setEnabled(false);
	  }
       }
       else
       {
          chromcheck.setSelected(false);
          chromcheck.setEnabled(false);
       }
    }


    //////////////////////////////////////////////////////////////////////////
    /**
     * Handles changes to changes in GO or cross-reference data sources
     */
    public synchronized void handlendbval()
    {
       if (xrefField.isEditable())
       {
          szuserxref = xrefField.getText();
       }
	
       if (goField.isEditable())
       {
          szusergann = goField.getText();
       }

       ndb = orgcb.getSelectedIndex();
       szorganismsourceval = GoAnnotations.organisms[ndb];
       String[] currfiles = null;

       int nitems = xrefcb.getItemCount();

       if (nitems == 1)
       {
          xrefcb.removeItemAt(0);
          for (int nindex =0 ; nindex < GoAnnotations.defaultxrefs.length; nindex++)
          {
     	     xrefcb.addItem(GoAnnotations.defaultxrefs[nindex]);
	  }
       }


       if ((GoAnnotations.xreforgfile[ndb].equals("")))
       {
	   
          if (ndb==1)
          {
	     //removing user provided option
	     xrefcb.setSelectedIndex(1);
          }
	  else
	  {
             xrefcb.setSelectedIndex(0);
	  }
       }
       else
       {	 
	   //System.out.println(ndb+"\t"+xreforgfile[ndb]);
	   xrefcb.setSelectedItem(GoAnnotations.organisms[ndb]);
       }


       //changed 7/5/09 to default to always blank for selecting organisms
       /*
       if (organismschrom[ndb].equals(""))
       {
          chromcb.setSelectedIndex(0);
       }
       else
       {
          chromcb.setSelectedItem(organismschrom[ndb]);
       }
       */
       chromcb.setSelectedIndex(0);
       handlechromval();

       if (ndb <= 1)
       {
          anncheck.setSelected(false);
	  anncheck.setEnabled(false);
       }
       else
       {
          File goannFile = new File(GoAnnotations.gannfile[ndb]);
	  if (goannFile.exists())
	  {
	     anncheck.setSelected(false);
	     anncheck.setEnabled(true);
	  }
	  else
	  {
	     anncheck.setSelected(true);
             anncheck.setEnabled(false);
          }
       }

       if (ndb == 0)
       {
          obocheck.setEnabled(true);
          obocheck.setSelected(false);
       }
       else if (ndb == 1)
       {
          obocheck.setSelected(false);
          obocheck.setEnabled(false);
       }
       else
       {
          File oboFile = new File(szgocategoryval);
       
          if (oboFile.exists())
          {
             obocheck.setEnabled(true);
             obocheck.setSelected(false);
          }
          else
          {
             obocheck.setSelected(true);
             obocheck.setEnabled(false);
          }
       }

       handlenxrefval();
       if (ndb>=1)
       {
          goField.setText(GoAnnotations.gannfile[ndb]);
	  goField.setEditable(false);
	  goLabelButton.setEnabled(false);
       }
       else
       {
          goLabelButton.setEnabled(true);
	  goField.setText(szusergann);
	  goField.setEditable(true);
       }
    }

    ///////////////////////////////////////////////////////////////
    /**
     * Manages the responds to actions on the interface
     */
    public void actionPerformed(ActionEvent e) 
    {
        Object esource = e.getSource();

        //Handle open button action.
        if (esource == viewButton)
	{
	    final String szfile = orig1Field.getText();
	    final JFrame fframe = this;
	    if ((new File(szfile)).exists())
	    {

               javax.swing.SwingUtilities.invokeLater(new Runnable() 
               {
                  public void run() 
                  {
		     DataTable newContentPane = new DataTable(fframe,szfile,false);
                     JFrame dtframe = new JFrame(szfile);
	             dtframe.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                     dtframe.setLocation(20,50);
                     newContentPane.setOpaque(true); //content panes must be opaque
                     dtframe.setContentPane(newContentPane);
                     //Display the window.
                     dtframe.pack();
                     dtframe.setVisible(true);
		  }
  	       });
            }
	    else
	    {
               javax.swing.SwingUtilities.invokeLater(new Runnable() 
               {
                  public void run() 
                  {
                      JOptionPane.showMessageDialog(fframe, "File '"+szfile+"' was not found.", 
                                  "Error", JOptionPane.ERROR_MESSAGE);
	           }
	        });
	    }    
	}
	else if (esource == clusteringmethodcb)
	{
	    handleclusteringmethod();
	}
	else if (esource == orgcb)
	{
	    handlendbval();
	} 
	else if (esource == xrefcb)
	{
	    handlenxrefval();
	}
	else if (esource == chromcb)
	{
	    handlechromval();
	}
        else if (esource == chromButton)
        {
            int returnVal = fc.showOpenDialog(this);

            if (returnVal == JFileChooser.APPROVE_OPTION) 
            {
               File file = fc.getSelectedFile();
               chromField.setText(file.getAbsolutePath());
            } 
        }
        else if (esource == xrefButton)
        {
            int returnVal = fc.showOpenDialog(this);

            if (returnVal == JFileChooser.APPROVE_OPTION) 
            {
               File file = fc.getSelectedFile();
               xrefField.setText(file.getAbsolutePath());
            } 
        } 
        else if (esource == orig1Button) 
        {
            int returnVal = fc.showOpenDialog(this);

            if (returnVal == JFileChooser.APPROVE_OPTION) 
            {
               File file = fc.getSelectedFile();
               orig1Field.setText(file.getAbsolutePath());
            } 
        } 
        else if (esource == extraButton) 
        {
            int returnVal = fc.showOpenDialog(this);

            if (returnVal == JFileChooser.APPROVE_OPTION) 
            {
               File file = fc.getSelectedFile();
               extraField.setText(file.getAbsolutePath());
            } 
        } 
        else if (esource == goLabelButton) 
        {
            int returnVal = fc.showOpenDialog(this);

            if (returnVal == JFileChooser.APPROVE_OPTION) 
            {
               File file = fc.getSelectedFile();
               goField.setText(file.getAbsolutePath());
            } 
        }
        else if (categoryIDButton == esource)
        {
            int returnVal = fc.showOpenDialog(this);

            if (returnVal == JFileChooser.APPROVE_OPTION) 
            {
               File file = fc.getSelectedFile();
               categoryIDField.setText(file.getAbsolutePath());
            } 
        }
        else if (esource == optionsButton)
	{
            theOptions.setLocation(this.getX()+5,this.getY()+100);
            theOptions.setVisible(true);
	} 	 
        else if (esource == repeatButton)
	{
            theRepeatList.setLocation(this.getX()+75,this.getY()+100);
            theRepeatList.setVisible(true);
	} 
        else if (esource == clusterAButton)
	{
           s1 = System.currentTimeMillis();
           String szcommand = e.getActionCommand();

           szorig1val =  orig1Field.getText();
	   szchromval = chromField.getText();
	   szxrefval = xrefField.getText();
           szorig2val = "";
           szgoval =  goField.getText();
           szgocategoryval =szGeneOntologyFileDEF;
           szextraval = extraField.getText();
           szcategoryIDval = categoryIDField.getText();

           szmaxmissingval = thespinnermaxmissing.getValue().toString();
           szexpressval = thespinnerexpress.getValue().toString();
           szfilterthresholdval =thespinnerfilterthreshold.getValue().toString();
           szlbval = thespinnerlb.getValue().toString();
           szalphaval = thespinneralpha.getValue().toString();
           szpercentileval = thespinnerpercentile.getValue().toString();
           szmaxchangeval = thespinnerChange.getValue().toString();
           sznumberprofilesval =  thespinnerProfile.getValue().toString();
           szcorrmodelval = thespinnercorrmodel.getValue().toString();
           szsamplegeneval = thespinnersamplegene.getValue().toString();
           szsamplepvalval = thespinnersamplepval.getValue().toString();
           szsamplemodelval = thespinnersamplemodel.getValue().toString();
           szmingoval = thespinnermingo.getValue().toString();
           szmingolevelval =  thespinnermingolevel.getValue().toString();
	   btakelog = lognormButton.isSelected();
           bspotincluded =spotcheck.isSelected();
	   badd0 = nonormButton.isSelected();
	   szevidenceval = evidenceField.getText().trim();
           sztaxonval= taxonField.getText().trim();
           bpontoval = pcheck.isSelected();
           bcontoval= ccheck.isSelected();
           bfontoval= fcheck.isSelected();
           brandomgoval= randomgoButton.isSelected();

           bmaxminval = maxminButton.isSelected(); 
	   ballpermuteval = permutecheck.isSelected();

           if (noneButton.isSelected())
	   {
               nfdr = 0;
	   }
           else if (fdrButton.isSelected())
	   {
               nfdr = 1;
	   }
           else
	   {
               nfdr = 2;
	   }
           balltime = theRepeatList.allButton.isSelected();


           System.out.println(szcommand); 
  
           this.setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR));

	   ntodownload = 0;
	   nexceptions = 0;
	   final boolean bgetxref = xrefcheck.isSelected();
	   final boolean  bgetgoann = anncheck.isSelected();
	   final boolean bgetchromval = chromcheck.isSelected();

           if (bgetgoann)
           {
	      synchronized (lockpd)
	      {
	         ntodownload++;
		 bdownloading[0] = true;
		 bexception[0] = false;
		 npercentdone[0]= 0;
	      }
  	      Runnable runpd= new Progressdownload(this,0,szgoval);
              (new Thread(runpd)).start();
	   }	
	      
	   if (bgetxref)
	   {
	      synchronized (lockpd)
              {
	         ntodownload++;
		 bdownloading[1] = true;
		 bexception[1] = false;
		 npercentdone[1]= 0;
	      }
	      Runnable runpd= new Progressdownload(this,1,szxrefval);
              (new Thread(runpd)).start();
	   }

	   if (bgetchromval)
	   {
              synchronized (lockpd)
	      {
	         ntodownload++;
	         bdownloading[2] = true;
	         bexception[2] = false;
	         npercentdone[2]= 0;
	      }
	      Runnable runpd= new Progressdownload(this,2,szchromval);
              (new Thread(runpd)).start();
	   }
   
           if (obocheck.isSelected())
	   {
	      synchronized (lockpd)
	      {
	         ntodownload++;
	         bdownloading[3] = true;
	         bexception[3] = false;
	         npercentdone[3]= 0;
	      }

	      Runnable runpd= new Progressdownload(this,3,szgocategoryval);
              (new Thread(runpd)).start();
	   }
	    
           final JFrame fframe = this;
           Runnable clusterrun = new Runnable()
           {
	      public void run()
	      {
	         clusterAButton.setEnabled(false);
	         try
                 {
		    clusterscript(szchromval,szxrefval,szorig1val,
                               szorig2val,szgoval,
                               szgocategoryval, szmaxmissingval,
                               szexpressval,szfilterthresholdval,
                               szlbval, szalphaval,
                               szpercentileval,szmaxchangeval,
                               sznumberprofilesval, szcorrmodelval, szsamplegeneval,szsamplepvalval,
                               szsamplemodelval, szmingoval,nfdr,szmingolevelval,
			       szextraval,balltime,theRepeatList.data,btakelog,
			       bgetchromval,bgetxref, bgetgoann,bspotincluded,badd0,szcategoryIDval,
			       szevidenceval,sztaxonval,bpontoval,
			       bcontoval,bfontoval,brandomgoval,bmaxminval,ballpermuteval);
		 }
                 catch(IllegalArgumentException iex)
	         {
                    final IllegalArgumentException fiex = iex;
		    iex.printStackTrace(System.out);

		    if (executeDialognf != null)
		    {
		       executeDialognf.setVisible(false);
		       executeDialognf.dispose();
		    }

                    javax.swing.SwingUtilities.invokeLater(new Runnable() 
                    {
                       public void run() 
                       {
                          JOptionPane.showMessageDialog(fframe, fiex.getMessage(), 
                                 "Error", JOptionPane.ERROR_MESSAGE);
		       }
	            });
	         }
                 catch (Exception ex)
                 {
	            final Exception fex = ex;

		    if (executeDialognf != null)
		    {
		       executeDialognf.setVisible(false);
		       executeDialognf.dispose();
		    }

                    javax.swing.SwingUtilities.invokeLater(new Runnable() 
                    {
                       public void run() 
                       {
                          JOptionPane.showMessageDialog(fframe, fex.toString(), 
                               "Exception thrown", JOptionPane.ERROR_MESSAGE);
		      
                          fex.printStackTrace(System.out);
	               }
	            });
		 }
	         clusterAButton.setEnabled(true);
                 fframe.setCursor(Cursor.getPredefinedCursor(Cursor.DEFAULT_CURSOR)); 
	      }
	   };
	   (new Thread(clusterrun)).start();
	}
        else if (esource == openButton)
        {
           int returnVal = ST.fc.showOpenDialog(this);
	   String szMissing = "";
           StringBuffer szMissingBuf = new StringBuffer();

           if (returnVal == JFileChooser.APPROVE_OPTION) 
	   {  
	      BufferedReader br = null;
              try
	      {
                 File file = ST.fc.getSelectedFile();
		 String szfilename = file.getPath();
		 if (file.exists())
		 {
		    ST.initializeDefaults();
  	            parseDefaults(szfilename);
		    updateSettings();
		 }
		 else
		 {
                    JOptionPane.showMessageDialog(this, "The file "+szfilename+" was not found", 
                               "File Not Found", JOptionPane.ERROR_MESSAGE);
		 }
	      }
              catch(IllegalArgumentException iex)
              {
	         updateSettings();
                 final IllegalArgumentException fiex = iex;
	         final JFrame fframe = this; 
                 javax.swing.SwingUtilities.invokeLater(new Runnable() 
                 {
                    public void run() 
                    {
                       JOptionPane.showMessageDialog(fframe, fiex.getMessage(), 
                                "Exception thrown", JOptionPane.ERROR_MESSAGE);
	            }
	         });
	      }
              catch (IOException ex)
	      {
                 JOptionPane.showMessageDialog(this, ex.getMessage(), 
                               "Exception thrown", JOptionPane.ERROR_MESSAGE);
                 ex.printStackTrace(System.out);
              }
	      finally
	      {
		 if (br != null)
		 {
                     try
		     {
		        br.close();
		     }
		     catch (IOException ex)
		     {
                        JOptionPane.showMessageDialog(this, ex.getMessage(), 
                               "Exception thrown", JOptionPane.ERROR_MESSAGE);
                        ex.printStackTrace(System.out);
		     }   
		 }
	      }	     
	   }
       }
       else if (esource == infoButton)
       {
           String szMessage = "This is version 1.3.9 of the Short Time-series Expression Miner (STEM).\n\n"+
                       "The Short Time-series Expression Miner (STEM) was developed by Jason Ernst, "+
                       "Dima Patek, and Ziv Bar-Joseph. " +
                       "Any questions or bugs found should "+
	               "be emailed to Jason Ernst (jernst@cs.cmu.edu).";
   
	   Util.renderDialog(this,szMessage,50,100,"Information");
       }
       else if (
		(randomgoHButton==esource)||
		(evidenceHButton==esource)||
                (ontoHButton==esource)|| 
                 ( taxonHButton==esource)|| 
                 (categoryIDHButton == esource) ||
                 (orig1HButton == esource) ||
                 (orig2HButton == esource) ||
                 (advancedHButton == esource) ||
                 (goLabelHButton == esource) ||
                 (numberProfileHButton == esource) ||
                 (changeHButton == esource) ||
                 (maxmissingHButton == esource) ||
                 (filterthresholdHButton == esource) ||
                 (expressHButton == esource) ||
                 (corrmodelHButton == esource) ||
                 (alphamodelHButton == esource) ||
                 (lbHButton == esource) ||
                 (percentileHButton == esource) ||
                 (samplegeneHButton == esource) ||
                 (samplemodelHButton == esource) ||
                 (samplepvalHButton == esource)||
                 (mingoHButton == esource)||
		 (mingolevelHButton == esource)||
		 (methodHButton == esource)||
 		 (clusteringmethodHButton== esource)||
		 (logHButton == esource)||
		 (downloadlistgoHButton == esource)||
		 (chromcbHButton == esource)||
		 (chromHButton == esource)||
		 (xrefHButton == esource)||
		 (presetsHButton == esource)||
		 (executeHButton == esource)||
		 (xrefsourceHButton == esource)||
		 (viewHButton == esource)||
		 (spotHButton == esource)||
 		 (filterchoiceHButton == esource)||
		 (permuteHButton == esource)||
		(extraHButton == esource))
	{
            makeHelpDialog((Component) esource);
	}
    }

    ///////////////////////////////////////////////////////////////////////////////
    /**
     * Inner class controlling the downloading of external files
     */
    public class Downloadfile implements Runnable
    {
       int ntype;
       String szfile;

       Downloadfile(int ntype, String szfile)
       {
          this.ntype = ntype;
	  this.szfile = szfile;
       }

       public void run()
       {
          String szurl="";
          final String szfilef = szfile;

          if (ntype == 0)
          { 
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
	  }
          else if (ntype == 1)
          {
	     szurl = EBIURL;
	     if (szfile.equals("chicken.xrefs.gz"))
	     {
	        szurl +="CHICKEN/";
	     }
	     else if (szfile.equals("cow.xrefs.gz"))
	     {
	        szurl +="COW/";
	     }
             else if (szfile.equals("human.xrefs.gz"))
             {
                szurl +="HUMAN/";
             }
             else if (szfile.equals("mouse.xrefs.gz"))
	     {
	        szurl += "MOUSE/";
             }
             else if (szfile.equals("rat.xrefs.gz"))
	     {
                szurl += "RAT/";
             }
  	     else if (szfile.equals("arabidopsis.xrefs.gz"))
	     {
	        szurl +="ARABIDOPSIS/";
	     }
             else if (szfile.equals("zebrafish.xrefs.gz"))
	     {
       	        szurl +="ZEBRAFISH/";
             }
             szurl += szfile;
	  } 
          else if (ntype == 2)
	  {
	     szurl = "";
	  }
          else 
          {
	     szurl = "http://www.geneontology.org/ontology/gene_ontology.obo";
          }

          final String szurlf = szurl;

          getFile(szurlf,szfilef,ntype);
	       
	  synchronized(lockpd)
          {
	      ntodownload--;
	      bdownloading[ntype] = false;
	      lockpd.notifyAll();
          }
       }
   }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Inner class rendering the progress download display
     */
   public class Progressdownload implements Runnable
   {
       String szfile;
       int ntype;
       JFrame theframe;
	
       Progressdownload(JFrame theframe,int ntype,String szfile)
       {
          this.theframe =theframe;
          this.ntype = ntype;
          this.szfile = szfile;
       }

       public void run()
       {
          //launch download thread
	  Runnable rundf= new Downloadfile(ntype,szfile);
	  (new Thread(rundf)).start();
          final int noffsetxf = 250;
          final int noffsetyf;
          final int ntypef = ntype;
          if (ntypef == 0)
          {
             noffsetyf = 50;
          }
          else if (ntypef == 1)
          {
	     noffsetyf = 150;
          }
          else if (ntypef == 2)
          {
             noffsetyf = 250;
          }
	  else
	  {
             noffsetyf = 350;
          }

          final JProgressBar thebar = new JProgressBar(0,100);
          final JDialog progressDialog;
          if (ntypef==0)
          {
             progressDialog = new JDialog(theframe,"Gene Annotation File Download Progress");
	  }
	  else if (ntypef == 1)
	  {
	     progressDialog = 
	         new JDialog(theframe,"Cross Reference File Download Progress");
	  }	
	  else if (ntypef == 2)
	  {
	      progressDialog = 
		  new JDialog(theframe,"Biomart File Download");
          }	
	  else 
	  {
	      progressDialog= new JDialog(theframe,"Ontology File Download Progress");
	  }
          progressDialog.setCursor(Cursor.getPredefinedCursor(Cursor.WAIT_CURSOR)); 

          javax.swing.SwingUtilities.invokeLater(new Runnable() 
          {
	     public void run() 
             {
                synchronized(lockpd)
                {  
		   npercentdone[ntype] = 0;
		   thebar.setValue(0);
                   if (!bexception[ntype])
	           {
                      Container theprogressDialogPane = progressDialog.getContentPane();
	              progressDialog.setSize(400,75);
	              thebar.setSize(400,75);
                      thebar.setStringPainted(true); 
                      progressDialog.setBackground(Color.white);
                      theprogressDialogPane.setBackground(Color.white); 
                      theprogressDialogPane.add(thebar);
                      progressDialog.setLocation(theframe.getX()+noffsetxf,
                                                     theframe.getY()+noffsetyf);       
                      progressDialog.setDefaultCloseOperation(DISPOSE_ON_CLOSE);
                      progressDialog.setVisible(true);
		   }
		}
	     }
	   });

           int nlastpercentdone = 0;

	   synchronized(lockpd)
           {  
	      bdownloading[ntype] = true;
	      while (bdownloading[ntype])
	      {
	         try
                 {
		    lockpd.wait();
		 }
		 catch (InterruptedException ex)
                 {
                    ex.printStackTrace(System.out);
                 }
	         thebar.setValue(npercentdone[ntype]);
	      }
	   }

	   if  ((!bexception[ntype])&&(ntype == 0))
	   {
	       anncheck.setEnabled(true);
	       anncheck.setSelected(false);
	   }
	   else if  ((!bexception[ntype])&&(ntype == 1))
	   {
	       xrefcheck.setEnabled(true);
	       xrefcheck.setSelected(false);
	   }
	   else if  ((!bexception[ntype])&&(ntype == 2))
	   {
	       chromcheck.setEnabled(true);
	       chromcheck.setSelected(false);
	   }
	   else if  ((!bexception[ntype])&&(ntype == 3))
	   {
	       obocheck.setEnabled(true);
	       obocheck.setSelected(false);
	   }
	   progressDialog.setVisible(false);
	   progressDialog.dispose();
           progressDialog.setCursor(Cursor.getPredefinedCursor(Cursor.DEFAULT_CURSOR)); 
       }      
    }
    
    //////////////////////////////////////////////////////

    /**
     * Renders help information for the various help buttons on the interface
     */
    public void makeHelpDialog(Component esource)
    {
       String szMessage = "";
       JTextArea textArea = null;

        if (esource == permuteHButton)
        {
	    szMessage = "If the box 'Permutation Test Should Permute Time Point 0' "+
                        "is checked then the permutation test permutes all time points including "+
                        "time point 0 when computing the expected number of genes assigned to a profile.  "+
                        "In this case STEM  "+
                        "finds profiles with significantly more genes assigned "+
                        "than expected if all the input columns "+
                        "had been randomly reordered.  If this box is not checked the permutation test permutes all "+
                        "time points except for time point 0.  In this case STEM finds profiles with more genes "+
                        "assigned than expected if all the columns except for the first column "+
		        "had been randomly reordered.  Note that if 'No normalization/Add 0' was selected "+
                        "on the main input interface "+
                        "the column of added 0s is considered the first input column.\n\n"+
                        "Permuting time point 0 "+
                        "is generally preferred since only this test takes into account significant changes that "+
	      	        "occur between time point 0 and the immediate next time point.  "+
                        "However in some cases based on experimental design a gene's expression value before transformation at time point 0 "+
                        "is expected to be known "+
                        "more accurately than the other time points, and because of this asymmetry as "+
                        "explained below not permuting time point 0 can also be useful.\n\n"+
                        "The time point 0 expression value before transformation can be "+
                        "known more accurately than other time points "+
                        "in a two channel experiment where the "+
                        "time point 0 sample is used as the reference sample, or in a "+
                        "single channel experiment where extra repeats were done for time point 0.  "+
                        "In these experiments there is a lower variance in a gene's time point 0 expression value than at "+
                        "other time points.  One could thus expect for these time series experiments, that "+
                        "profiles centered around 0 with high variance will be more likely "+
                        "to be considered significant in a permutation test that permutes this low variance time point 0.  "+
                        "A permutation test that does not permute time point 0 can be useful here since "+
                        "profiles found to be significant under this test are significant independent "+
                        "of the time point 0 expression value being known more accurately than that of the other time points.  "+
                        "In practice the set profiles found to be "+
		        "significant by either test will usually be similar.";
	    Util.renderDialog(theOptions,szMessage,50,100);
	}
        else if (esource == filterchoiceHButton)
	{
	    szMessage = "The 'Change should be based on parameter' defines how change is defined in the context "+
                        "of gene filter.  "+
                        "If 'Maximum\u2212Minimum' option is selected a gene will be filtered if the maximum "+
                        "absolute difference between "+
                        "the values of any two time points, not necessarily consecutive, "+
                        "after transformation is less than the value of "+
                        "the 'Minimum Absolute Expression Change' parameter.  "+
                        "If 'Difference from 0' is selected a gene "+
                        "will be filtered if the absolute expression change from time point 0 at all time points is less than the "+
                        "value of the 'Minimum Absolute Expression Change' parameter.\n\n"+
                        "Formally suppose (0,v_1,v_2,...,v_n) is the expression level of a gene "+
                        "after transformation and "+
                        "let C be the value of the 'Minimum Absolute Expression Change'.  "+
                        "If the 'Maximum-Minimum' option is selected a gene will be filtered if "+
                        "max(0,v_1,v_2,...v_n)\u2212min(0,v_1,v_2,...,v_n)<C.  If the "+
		        "'Minimum Absolute Expression Change' option is selected "+ 
                        "the gene will be filtered "+
	 	        "if max(0,|v_1|,|v_2|,...,|v_n|)<C.\n\n"+
                        "Only the 'Maximum\u2212Minimum' option guarantees that the "+
                        "same set of genes would be filtered for any permutation of the time points.  "+
		        "For the 'Difference from 0' this is in general not true, in this case the permutation "+
                         "test is based on the set of genes passing filter under the original order of time points.";
           Util.renderDialog(theOptions,szMessage,50,100);
	}
	else if (esource  ==ontoHButton)
	{
           szMessage =
                 "These three checkboxes allow one to filter annotations that are not "+
                 "of the types checked.  These three checkboxes only apply if the annotations are in the official "+
                 "15 column GO format, in which case the annotation type is determined by "+
                 "the entry in the 'Aspect' field (Column 9).  "+
                 "An entry of 'P' in the 'Aspect' field means the annotation is of type "+
                 "'Biological Process', an entry of 'F' "+ 
                 "means the annotation is of type 'Molecular Function', and an entry of 'C' means the annotation "+
	         "is of type 'Cellular Component'.";
           Util.renderDialog(theOptions,szMessage,50,100);
	}
	else if (esource  == taxonHButton)
	{
           szMessage = 
                 "Some annotation files contain annotations for multiple "+
                  "species, and it might be desirable to use annotations only for certain species.  "+
                  "To use only annotations for "+
                  "certain species enter the taxon IDs for the desired species delimited "+
                  "by either commas (','), semicolons (';'), or pipes ('|').  "+
                  "If this field is left empty, then any species is assumed to be acceptable.  More information "+
                  "about taxonomy codes and a search function to find the taxon code for a species can be found at "+
                  "http://www.ncbi.nlm.nih.gov/Taxonomy/.  Note that this parameter only applies when "+
                  "the annotations are in the official 15 column format.  "+
                  "The taxonomy ID in the annotation file is in column 13 of the file, and the taxon "+
                  "IDs entered in this parameter field must match the entry in column 13 or match after "+
                  "prepending the string 'taxon:' to the ID.  For example to use only annotations "+
	          "for a 'Homo sapien' the string '9606' can be used."; 

           Util.renderDialog(theOptions,szMessage,50,100);
	}
	else if (esource  == evidenceHButton)
	{
           szMessage = 
               "This field takes a list of unacceptable evidence codes for "+
               "gene annotations delimited by either a comma (','), semicolon (';'), "+
               "or pipe ('|').  If this field is left empty, then "+
               "all evidence codes are assumed to be acceptable.  Evidence code symbols are "+
               "IEA, IC, IDA, IEP, IGI, IMP, IPI, ISS, RCA, NAS, ND, TAS, and NR.  "+
               "Information about GO evidence codes can be found at "+
               "http://www.geneontology.org/GO.evidence.codes.shtml.  Note that this field only applies "+
               "if the gene annotations are in the official 15 column GO annotation format.  "+
               "The evidence code is the entry in column 7.  "+
               "For example to exclude the annotations that were inferred from electronic annotation "+
	       "or a non-traceable author statement the field should contain IEA;NAS.";

           Util.renderDialog(theOptions,szMessage,50,100);
	}
        else if (esource == viewHButton)
	{
	   szMessage = "Pressing the 'View Data File' button opens a table with "+
                       "the contents of the file listed under 'Data File'.";
           Util.renderDialog(this,szMessage,50,100);
	}
        else if (esource == categoryIDHButton)
	{
	    szMessage = "This file, which is optional, specifies a mapping between category IDs and names.  "+
                        "The first column contains category IDs while the second column contains category names "+
                        "corresponding to the category ID in the first column.  Note that the category names for "+
                        "official Gene Ontology (GO) categories are included in the 'gene_ontology.obo' file "+
                        "and thus do not need to be included here. "+
                        "This file is rather intended to define names of additional gene sets "+
                        "that are not part of GO, but will be included in a GO analysis.  "+
                        "If no mapping is made between a category ID and category name, then the category ID "+
                        "is used in place of the category name.\n\n"+
                        "SAMPLE FILE:\n"+
		        "ID_A	Category A\n"+
		        "ID_B	Category B\n"+
	       		"ID_C	Category C\n";

           Util.renderDialog(theOptions,szMessage,50,100);
	}

        if (esource == spotHButton)
	{
	   szMessage = "Each entry in the data file is associated with a unique identifier called a spot ID.  "+
                 "This identifier is different than the genes symbols which need not be unique.  "+
                 "Spot IDs can either be included in the data file as the first column or not included and then "+
                 "automatically generated.\n"+
                 "*If the 'Spot IDs included in the data file' box is checked then the first column of "+
                 "the data file contains the spot IDs "+
                 "and the second column contains gene symbols.\n"+
                 "*If the 'Spot IDs included in the data file' box is unchecked then  the first column of the data "+
                 "file contains gene symbols and the spot IDs are "+
                 "automatically generated for each entry in "+
                 "the data file.  "+
                 "Spot IDs are automatically generated as sequential integers starting at 0 "+
	         "appended to the string \"ID_\".\n";

	   Util.renderDialog(this,szMessage,50,100);
	}
	else if (esource == logHButton)
        {
          szMessage = 
                      "All time series will be transformed so that the "+
                       "time series starts at 0.  "+
                      "This can be done in one of three ways based on the option selected to the left.  "+
                      "Given a time series vector of values for a gene (v_0,v_1,...,v_n) the options are:\n"+
                      "1.  'Log normalize data'  \u2212  the vector "+
                      "will be transformed to (0,log\u2082(v_1)\u2212log\u2082(v_0),...,log\u2082(v_n)\u2212log\u2082(v_0)).  "+
                      "Note that any values which are 0 or negative will be treated as missing.\n"+
                      "2.  'Normalize data'  \u2212  the vector "+
                      "will be transformed to (0,v_1\u2212v_0,...,v_n\u2212v_0)\n"+
                      "3.  'No normalization/add 0'  \u2212  a 0 will be inserted transforming the vector to "+
                      "(0,v_0,v_1,...,v_n)\n\n"+
                      "*If the data is not already in log space as often is the case if it is "+
                      "from an Oligonucleotide array, then the "+
                      "'Log normalize data' should be selected.\n"+
                      "*If the data is already in log space as is often the case if the data is "+
                      "from a two channel cDNA array and "+
                      "a time point 0 experiment was "+
                      "conducted, then the 'Normalize data' option should be selected.\n"+
                      "*If the data is already in log space and no time point 0 experiment was "+
                      "conducted, then the 'No normalization/add 0' option should be selected."; 
	  Util.renderDialog(this,szMessage,50,100);
       }
       else if (esource == orig1HButton)
       {
	  szMessage = "This entry specifies a file that contains gene expression data.  "+
                      "A data file includes gene symbols, data values, and optionally spot IDs. "+
                      "Spot IDs uniquely identify an entry.  If spot IDs are included in the data file the field "+
                      "'Spot IDs included in the data file' must be checked, otherwise the field must be unchecked and "+
                      "IDs for each entry in the data file will be generated.  "+
                      "While spot IDs must be unique, the same gene symbol can appear multiple times in the data file.\n\n"+
                      "The file has the following "+
                      "formatting restrictions:\n"+
                      "* The first line contains the column headers delimited by tabs.\n"+
                      "* The remaining lines contain the spot IDs (optionally), "+
                      "gene symbols, and then data delimited by tabs.\n"+
	              "* If the 'Spot IDs included in the data file' box is checked, then the first column "+
                      "contains the spot IDs and each spot ID must be unique.\n"+
                      "* The next column, or the first column if the 'Spot IDs included in the data file' "+
                      "box is unchecked, contains the gene symbols.  If there is no gene symbol associated with a given spot,"+
                      " then the field can either contain a \"0\" or no entry.\n"+
                      "* In either the spot or gene field there can be multiple symbols listed delimited by "+
                      "either a pipe ('|'), comma (','), or a semi-colon (';').  "+
                      "For the purposes of gene annotations just one symbol needs to match an annotation, while "+
                     "for the purpose of determining if two spots or genes are the same the entire entry must match.\n"+
                      "* If multiple spots correspond to the same gene, "+
                      "then the data for that gene will be combined using the median values after "+
                      "normalization.\n"+
                      "* The remaining columns contain the expression values in sequential order of time for the gene.\n"+
                      "* If a value is missing between two time points then the field should be left empty giving "+
                      "two tabs between the non-missing values.\n\n"+
	              "SAMPLE FILE with spot IDs included:\n"+
                      "Spot	Gene	0h	1h	3h	6h	12h\n"+
                      "ID_1	ZFX	-0.027	0.158	0.169	0.193	-0.165\n"+  
                      "ID_2	ZNF133	0.183	-0.068	-0.134	-0.252	0.177\n"+
                      "ID_3	USP2	-0.67	-0.709	-0.347	-0.779	-0.403\n"+
                      "ID_4	DSCR1L1	-0.923	-0.51	-0.718	-0.512	-0.668\n\n"+
	              "SAMPLE FILE without spot IDs included:\n"+
                      "Gene	0h	1h	3h	6h	12h\n"+
                      "ZFX	-0.027	0.158	0.169	0.193	-0.165\n"+  
                      "ZNF133	0.183	-0.068	-0.134	-0.252	0.177\n"+
                      "USP2	-0.67	-0.709	-0.347	-0.779	-0.403\n"+
                      "DSCR1L1	-0.923	-0.51	-0.718	-0.512	-0.668";

	  Util.renderDialog(this,szMessage,50,100);
       }
       else if (esource == extraHButton)
       {
           szMessage = "This file is optional.  If included any genes listed in the file "+
                     "will be considered part of the initial base set of genes during a GO analysis in  "+
                     "addition to any genes included in the data file. "+
                     " Using this file thus allows one to filter "+
                     "genes from the data set by a criteria not implemented in STEM by excluding them from the "+
                     "data file, but still include the "+
                     "filtered genes as part of the "+
                     "initial base set of genes during the GO analysis by including them in this file.  "+
                     "If genes appear in both this file and the data file the gene will only "+
                     "be added to the base set once.  "+
                     "The format of this file is the same as a data file, except including the expression values is "+
                     "optional and if included they will be ignored.  "+
                     "As with a data file the first column will contain spot IDs "+
  	             "if the field 'Spot IDs included in the data file' is checked and the second column will contain "+
	             "gene symbols, otherwise the first column will contain gene symbols."; 

           Util.renderDialog(theOptions,szMessage,50,100);
       }
       else if (esource == advancedHButton)
       {
           szMessage =
                     "  Pressing the 'Advanced Options' button opens a dialog box with "+
                     "advanced options pertaining to filtering genes, selecting and assessing the "+
                     "statistical significance of model profiles, "+
                     "clustering the significant model profiles, "+
	             "gene annotations, and the Gene Ontology (GO) enrichment analysis.  "+
                     "The options under the 'Model Profiles' and 'Clustering Profiles' tabs "+
	             "are only relevant if the STEM clustering method is selected.";
	   Util.renderDialog(this,szMessage,50,100);

       }
       else if (esource == randomgoHButton)
       {
	   szMessage = 
                  "This parameter controls the correction method for actual size based GO enrichment.  "+
                  "Expected size based p-values are always "+
                  "corrected using a Bonferroni correction.  "+
                  "The parameter value can either be 'Bonferroni' or 'Randomization'.  "+
                  "If 'Bonferroni' is selected then a Bonferroni correction is applied where the uncorrected "+
                  "p-value is divided by the number of categories meeting the minimum "+
                  "'Minimum GO level' and 'Minimum number of genes' constraints.  "+
                  "If 'Randomization' is selected the corrected p-value is computed "+
                  "based on a randomization test where random samples "+
                  "of the same size of the set being analyzed is drawn.  "+
                  "The number of samples is specified by the parameter "+
                  "'Number of samples for multiple hypothesis correction'.  "+  
                  "The corrected p-value for a p-value, r, is the proportion of random "+
                  "samples for which there is enrichment for any GO category with a p-value less than r.  "+  
	         "A Bonferroni correction is faster, but a randomization test leads to lower p-values.";

            Util.renderDialog(theOptions,szMessage,50,100);
       }
       else if (esource == orig2HButton)
       {
          szMessage = "     Clicking on the 'Repeat Data' button brings up a dialog box "+
                      "to specify repeat data files of the experiment.  Repeat data files "+
                      "are optional.  The format of repeat data is the same as the original data.  "+
                      "If included the repeat data values will be averaged using the median with the "+
                      "original experiment values.  "+
                      "Repeat data can either represent repeat measurements taken concurrently with the "+
                      "original experiment, or distinct full repeat experiments taken at different time periods.  "+
                      "If the data is of the latter type genes will be filtered that do not display a consistent "+
                      "expression profile between repeats based on the 'Minimum Correlation between Repeats' "+
                     "parameter under the 'Filtering' panel on the advanced options menu.  "+
                      "Also if the repeat data is from different time periods, then the clustering of model profiles "+
                      "can be adjusted using noise estimates from the repeat experiments based on "+
                      "the 'Minimum Correlation' parameter of the 'Clustering' panel on the advanced options menu.  "+
             "If the button is yellow then there is currently repeat data loaded, otherwise the button is gray.";

	  Util.renderDialog(this,szMessage,50,100);
       }
       else if (esource == goLabelHButton)
       {
	   szMessage = "This file contains the Gene Ontology (GO) annotations of genes.  "+
                       "The file can be in one of two formats:\n\n"+
                       "1.  The file can be in the official 15 column GO Annotation format "+
                       "described at http://www.geneontology.org/GO.annotation.shtml#file.  In this case any entry in the "+
                       "DB_Object_ID (Column 2), DB_Object_Symbol (Column 3), DB_Object_Name (Column 10), or "+
                       "DB_Object_Synonym (Column 11) fields matching a spot ID or gene symbol in the data set "+ 
                       "will be annotated as belonging to GO ID (Column 5).  "+
                       " If the entry in the 'DB_Object_Symbol' "+ 
                       "contains an underscore ('_'), then the portion of the entry before "+
                       "the underscore will also be annotated as "+
                       "belonging to the GO category since under some naming conventions "+
                       "the portion after the underscore is a symbol "+
                       "for the database that is not specific to the gene.  "+
                       "The 'DB_Object_Synonym' column may have multiple symbols delimited by either "+
                       "a semicolon (';'), comma (','), or a pipe ('|') symbol and all will be "+
                       "annotated as belonging to the GO category in Column 5.  "+
                       "Note that the exact content of the 'DB_Object_ID', 'DB_Object_Symbol', "+
                       "'DB_Object_Name', and 'DB_Object_Synonym' varies between annotation source, "+ 
                       "consult the README files available at http://www.geneontology.org/GO.current.annotations.shtml "+
	       "to find out more information about the content of these fields for a specific annotation source.\n\n"+

                       "2.  Alternatively the file can have two columns where the first column contains gene symbols "+
                       "or spot IDs and the second column contains annotations of the genes in the first column. "+
                       "The two columns are delimited by a tab.  "+
                       "Gene symbols and GO annotations can be delimited by "+
                       "either a semicolon (;), comma (','), or a pipe (|).\n\nNote:\n"+
                       "*If a gene is listed as belonging to a certain GO category that is a sub-category"+
                       " of other categories in the GO hierarchy, it is not necessary to also explicitly list "+
                       "its super-categories.\n"+
                        "*If the same gene appears on multiple lines the union of "+
                        "annotation terms is taken.\n"+
                        "*The file can either be in plain text or "+
                        "a gzipped version of a plain text file in the required format.\n\n"+ 
                       "Sample file of two column format:\n"+
	               "ZFX	GO:0003677;GO:0003713;GO:0008270;GO:0030528;GO:0046872;GO:0006355;GO:0005634\n"+
                       "ZNF133	GO:0003700;GO:0008270;GO:0006355;GO:0005634\n"+
		       "USP2	GO:0004197;GO:0004221;GO:0016787;GO:0006511";

           Util.renderDialog(this,szMessage,50,100);
       }
       else if (esource ==  numberProfileHButton)
       {
	   if (nclusteringmethodcb==0)
	   {
              szMessage = 
                       "     This parameter specifies the maximum number of model profiles that can be selected from the "+
                       "candidate model profiles.  The candidate model profiles are profiles which start at 0 "+
                       "and increase or decrease an integral number of units that is less than or equal to the maximum "+
                       "unit change between time points.  The constant 0 profile is excluded from the set of candidate model "+
                       "profiles.  In total the number of candidate model profiles is \n"+
                       "(2*(maximum unit change)+1)^((number of time points)-1)-1.\n"+
                       "The number of model profiles selected can also be indirectly restricted by placing an upper bound "+
	               "on the maximum correlation between any two model profiles under the model profiles panel on the "+
                       "advanced options menu.  "+
                       "If this parameter is 0, then there is no hard bound on the number of model profiles.  "+
                       "In the case the number of profiles will be determined by the number of candidate profiles and "+
                       "the maximum correlation between model profiles.\n\nThe model profiles selected are selected to "+
	               "be distinct from each other and are also representative of the entire set of candidate profiles.";
	   }
	   else
	   {
	       szMessage = " This parameter specifies the number of clusters, K, that the K-means algorithm should produce";
	   }

	   Util.renderDialog(this,szMessage,50,100);
       }
       else if (esource == changeHButton)
       {
	   if (nclusteringmethodcb==0)
	   {
              szMessage =  
                       "     This parameter specifies the maximum number of units a model "+
                       "profile may change between time "+
                       "points.  A model profile between two consecutive time points can either stay constant, or "+
                       "increase or decrease an integral number of units up to this parameter value.";
	   }
	   else
	   {
             szMessage = "  This parameter specifies the number of times the K-means algorithm should be ran "+
                         " each time with a different random start.  "+
                         "Only the best scoring clustering will be returned.  Increasing this parameter could "+
		         "give a slightly better clustering at a cost of slightly greater execution time.";
	   }
	   Util.renderDialog(this,szMessage,50,100);

       }
       else if (esource == maxmissingHButton)
       {
	   szMessage = "     This parameter specifies the maximum number of missing values that are allowed for a gene.  "
                      +"A gene will be filtered if the number of time points for which there are no readings for the gene is "
	              +"greater than this parameter.  A gene will also be filtered if its expression value at the first time "
                      +"point is missing and log normalize data or normalize data was selected as the data transformation.";                    

         Util.renderDialog(theOptions,szMessage,50,100);
       }
       else if (esource == filterthresholdHButton)
       {
           szMessage = "     This parameter only applies if there is repeat data over different time periods.  "+
                        "In the case of one repeat data set, the correlation value of the gene's expression level "+
                        "between the original and repeat must have "+
                        "a correlation value above this parameter, otherwise the gene will be filtered.  "+
                        "In the case of multiple repeat sets, the mean pairwise correlation over all data sets "+
	                "must have a correlation value above this parameter, otherwise the gene will be filtered.";

           Util.renderDialog(theOptions,szMessage,50,100);
       }
       else if (esource == expressHButton)
       {

           szMessage = "  After transformation (Log normalize data, Normalize data, or No Normalization/add 0) "+
                       "if the absolute value of the gene's largest change "+
                       "is below this threshold, then the gene will be filtered.  If the "+
                       "'Change should be based on' parameter is set to 'Maximum\u2212Minimum' "+
                       "then change is defined as the largest difference in the gene's value between "+
                       "any two time points, not necessarily consecutive.  "+
                       "Alternatively if the 'Change should be based on' "+
	       "parameter is set to 'Difference from 0', then change is based on the largest change from time point 0.";

           Util.renderDialog(theOptions,szMessage,50,100);
       }
       else if (esource == corrmodelHButton)
       {
           szMessage = 
                 "This parameter specifies the value that the maximum correlation between any "+
                 "two model profiles must be below, and thus can be used to guarantee that "+
                 "two very similar profiles will not be selected.  "+
                 "Lowering this parameter could have the effect that the number "+
                 "of model profiles selected is less than the "+
                 "'Maximum Number of Model Profiles' even if more candidate model profiles are available.  "+
                 "This parameter's maximum value is 1, thus preventing two perfectly "+
	         "correlated model profiles from being selected.  ";

            Util.renderDialog(theOptions,szMessage,50,100);
       }
       else if (esource == alphamodelHButton)
       {
           szMessage = 
                  "     The significance level at which the number of genes assigned to a model profile compared "+
                  "to the expected number of genes assigned should be considered significant.  If the correction "+
                  "method for multiple hypothesis testing is set to 'Bonferroni' then this parameter is the significance level "+
	          "before applying a Bonferroni correction.  If the correction method parameter is set to 'False Discovery "+
                  "Rate' this parameter is the false discovery rate.  If correction method parameter is 'none' "+
                  "then this parameter is the uncorrected significance level.";

           Util.renderDialog(theOptions,szMessage,50,100);
       }
       else if (esource == lbHButton)
       {
           szMessage = "     STEM groups statistically significant model profiles which are similar "+
                      "to each other into clusters of "+
                      "profiles.  Any two model profiles assigned to the same cluster of "+
                      "profiles must have a correlation above this parameter's value.  "+
                       "Increasing this value will lead to more clusters with fewer model profiles "
                      +"in each cluster while decreasing the value will lead to fewer clusters "+
                      "with more model profiles in each cluster.";

           Util.renderDialog(theOptions,szMessage,50,100);
       }  
       else if (esource == percentileHButton)
       {
           szMessage =
              "If there is repeat data selected to be from 'Different time periods', "+
              "then this parameter specifies that any two model profiles assigned to "+
              "the same cluster of profiles must have a correlation in their "+
              "expression greater than the correlation of this percentile in the distribution of "+
              "gene expression correlations between the repeats.  For instance if "+
              "this parameter value is 0.5, then any two model profiles assigned to "+
              "the same cluster will have a correlation greater than the median "+
              "correlation of gene expression correlations between the "+
              "repeats.  This parameter allows clustering of model profiles to be dependent "+
              "on the noise in the data.  If the 'Minimum Correlation' parameter is set to -1, "+
              "then only the 'Minimum Correlation Percentile' "+
              "parameter value will influence the clustering of model profiles.  "+
              "Similarly if the 'Minimum Correlation Percentile' parameter is set to 0, "+
              "then only the 'Minimum Correlation' "+
	      "parameter value will influence the clustering of model profiles.";

           Util.renderDialog(theOptions,szMessage,50,100);
       }
       else if (esource == samplegeneHButton)
       {
	   szMessage = "     This parameter specifies the number of permutations of time points that should be "
                      +"randomly selected for each gene when computing the expected number of genes assigned "
                      +"to each of the model profiles.  If this parameter is 0, then all permutations are used.  "+
                       "Increasing the number of permutations will lead to slightly greater accuracy at the expense "+
	               "of greater execution time.";

           Util.renderDialog(theOptions,szMessage,50,100);
       }
       else if (esource == samplemodelHButton)
       {
           szMessage =
                      "   Candidate model profiles are non-constant profiles which start at 0 and increase or "+
                      "decrease an integral number of units that is less than or equal to the value of the "+
                      "'Maximum Unit Change in Model Profiles between Time Points'.  "+
                      "If the number of candidate model profiles exceeds this parameter, "+
                      "then instead of explicitly generating all candidate model profiles a subset of candidate "+
                      "model profiles of this size will be randomly selected.  In most cases there will be no need "+
	              " to adjust this parameter.";

           Util.renderDialog(theOptions,szMessage,50,100);
       }
       else if (esource == samplepvalHButton)
       {
           szMessage = 
                "This parameter specifies the number "+
                "of random samples that should be made when computing multiple hypothesis "+
                "corrected enrichment p-values by a randomization test.  A randomization test "+
                "is used when the p-value enrichment is based on the actual size "+
                "of the set of genes and 'Randomization' is selected next to the "+
                "'Multiple hypothesis correction method for actual sized based enrichment' label. "+
                "The Bonferroni correction is always used when the p-value enrichment is based on the "+
                "expected size of the set of genes.  "+
                "Increasing this parameter will lead to more accurate corrected p-values "+
	        "for the randomization test, but will also lead to longer execution time to compute the values.";

           Util.renderDialog(theOptions,szMessage,50,100);
       }
       else if (esource == mingoHButton)
       {
           szMessage =
                 "For a category to be listed in a gene enrichment analysis table "+
                 "the number of genes in the set being analyzed that also belong to "+
	         "the category must be greater than or equal to this parameter.";

           Util.renderDialog(theOptions,szMessage,50,100);
       }
       else if (esource == mingolevelHButton)
       {
           szMessage =
               "   Any GO category whose level in the GO hierarchy is below this parameter will not be "+
               "included in the GO analysis.  The categories Biological Process, Molecular Function, and "+
               "Cellular Component are defined to be at level 1 in the hierarchy.  The level of any other "+
               "term is the length of the longest path to one of these three GO terms in terms of the number "+
               "of categories on the path.  This parameter thus allows one to exclude the most general GO categories.";
  
           Util.renderDialog(theOptions,szMessage,50,100);         
       }
       else if (esource == clusteringmethodHButton)
       {
	   szMessage = "This option specifies the clustering method to be used by STEM.  The clustering method can "+
                       "either be the STEM clustering method or the standard K-means clustering algorithm.  The "+
                       "STEM clustering method is a novel method for clustering short time series gene expression "+
                       "published in \n"+
                       "J. Ernst, G.J. Nau, and Z. Bar-Joseph. Clustering Short Time Series Gene Expression Data.\n"+
	               "Bioinformatics (Proceedings of ISMB 2005), 21 Suppl. 1, pp. i159-i168, 2005.";

             Util.renderDialog(theOptions,szMessage,50,100);     
       }
       else if (esource == methodHButton)
       {
           szMessage =
               "The significance level can be corrected for the fact that multiple profiles are being "+
               "tested for significance.  The correction can be a Bonferroni correction where "+
               "the significance level is divided by "+
               "the number of model profiles or the less conservative False Discovery "+
               "Rate control.  If 'none' is selected then no correction is made for the multiple "+
               "significance tests.  Note that this parameter for multiple test " +
	       "correction for model profiles is unrelated to the corrected p-values in a GO enrichment analysis. ";

           Util.renderDialog(theOptions,szMessage,50,100);
      }
      else if (esource == xrefsourceHButton)
      {
          szMessage =  
                    "A cross reference file specifies that two or more symbols for the same gene are equivalent.  "+
                     "This file is optional, but is useful "+
                     "in the case where annotation of genes in the annotation file use a different naming "+
	             "convention than the genes in the data file. "+
                     "With a cross reference file it is possible "+
                     "to match a gene in the data file with its annotation in the annotation file "+
                     "even if the symbol used for the gene in the data file "+
  	             "does not match the symbol used for the gene in the annotation "+
                     "file.\n\nUsing this menu a user can choose to "+
                     "either provide a cross reference file themselves ('User provided'), "+
                     "not to use a cross reference file ('No cross references'), or "+
                     "use a species specific cross-reference file "+
                     "provided by the European Bioinformatics Institute (EBI) "+
                     "available from "+
                     "ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/.\n\nNote that leaving the cross-reference field blank "+
	      "under 'User Provided' is equivalent to selecting 'No cross references'.";                   
	  Util.renderDialog(this,szMessage,50,100);
      }
      else if (esource == downloadlistgoHButton)
      {
	  szMessage = "If a box is checked then the corresponding file will be downloaded upon clicking execute "+
                      "and prior to analysis of the data. \n\n*The 'Annotation' field corresponds to the file listed in "+
                      "the 'Gene Annotation File' textbox.  Note that if the annotation source is user provided then "+
                      "the file cannot be downloaded automatically.  If the annotation file is not present locally it "+
                      "must be downloaded and thus the 'Annotation' check box will automatically be marked.  "+
                      "Gene annotation files are downloaded from "+
                      "http://www.geneontology.org/gene-associations/ unless it is "+
	      //"ftp://ftp.geneontology.org/go/gene-associations/ unless it is "+
                      "an EBI data source in which case it will be downloaded from "+
                      "ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/.\n\n"+
                      "*The 'Cross References' file corresponds to the file under the 'Cross References File' text "+
                      "box, this field is only enabled if the Annotation Source is an EBI species for which "+
                      "a cross reference file is provided.  If the cross reference file is not present locally "+
                      "on the computer it must be downloaded. Cross reference files are downloaded from "+
                      "ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/.\n\n"+
                      "*The 'Locations' checkbox corresponds to the data for the file under "+
                      "'Gene Location Source' which is obtained from http://www.biomart.org/biomart/martservice "+
                      "(NOTE: as of v1.3.9 automatic downloads of Gene Locations has been removed since the service is not working).\n\n"+
                      "*The 'Ontology' checkbox corresponds to the gene "+
                      "ontology specified in the gene_ontology.obo file.  It must be present if a non-user provided "+
	              "annotation source is selected, otherwise an attempt to download the file from "+
                      "http://www.geneontology.org/ontology/ will be made. \n\nCurrently:\n";
	   szxrefval = xrefField.getText();
           szgoval =  goField.getText();

	  String szontologydate ="";
	  String szgovaldate = "";
	  String szxrefdate = "";

	  File goannFile = new File(szgoval);
	  if (goannFile.exists())
	  {
             szgovaldate = "The gene annotation file ("+szgoval+") was last updated " 
                            +(new Date(goannFile.lastModified())).toString()+".\n";
	  }

  	  File xrefFile = new File(szxrefval);
          if (xrefFile.exists())
	  {
             szxrefdate = "Gene annotation file ("+szxrefval+") was last updated " 
                            +(new Date(xrefFile.lastModified())).toString()+".\n";
	  }

	  File oboFile = new File (szgocategoryval);
	  if (oboFile.exists())
	  {
             szontologydate = "The gene ontology file ("+szgocategoryval+") was last updated " 
                             +(new Date(oboFile.lastModified())).toString()+".";
	  }

  	  szMessage += szgovaldate+szxrefdate+szontologydate;

	  Util.renderDialog(this,szMessage,50,100);
      }
      else if (esource == chromHButton)
      {
	  szMessage = "This specifies the file containing the gene locations. It is editable when "+
	      "'User provided' is specified under 'Gene Location Source'.";
	  Util.renderDialog(this,szMessage,50,100);
      }
     else if (esource == chromcbHButton)
      {
	  szMessage = "This specifies the source of gene locations on chromosomes for the . "+
                     "chromosome viewer. If using the chromosome viewer is not desired then select "+
                      "'No Gene Locations'. A user can also specify locations using a GFF version 2 format "+
	               "file http://www.sanger.ac.uk/Software/formats/GFF/GFF_Spec.shtml by first selecting "+
	             "'User provided' and then specifying the file under 'Gene Location File'.";
	  Util.renderDialog(this,szMessage,50,100);
      }
      else if (esource == xrefHButton)
      {
	 szMessage = "     A cross reference file specifies that two or more symbols for the same gene are equivalent.  "+
                     "This file is optional, but is useful "+
                     "in the case where annotation of genes in the annotation file use a different naming "+
	             "convention than the genes in the data file, "+
                     "and thus with a cross reference file it is possible "+
                     "to match a gene in the data file with its annotation in the annotation file "+
                     "even when the symbol used in the data file "+
                     "does not appear in the annotation file.\n\n     Note that the cross reference file is only used "+
	             "to map between gene symbols and not a spot ID and a gene symbol.  Any symbols on the same line are"+
                     " considered to be equivalent where symbols are delimited by tabs, a pipe(|), a comma (',')"+
                     " or a semicolon (;).  The file can either be in plain text or "+
                     "gzipped version of a text file in this format. \n\n"+
                     "Sample cross reference file:\n"+
	             "GeneA	SymbolA\n"+
                     "GeneB	SymbolB\n"+
	             "GeneC	SymbolC";

	 Util.renderDialog(this,szMessage,50,100);
      }
      else if (esource == presetsHButton)
      {
	 szMessage = "The user can either select to provide their own gene annotation file ('User provided'), "+
                     "'No annotations', or use one of 35 gene annotation files available from the Gene Ontology "+
                     "Consortium.  More information about these 35 annotation files can be found here "+
                     "http://www.geneontology.org/GO.current.annotations.shtml and in the case of "+
                     "annotations from the European Bioinformatics Institue (EBI) also here http://www.ebi.ac.uk/GOA/.  "+
	             "If one of the predefined annotation files "+
                     "is selected then the annotation field will automatically be filled in and the option will be "+
                     "available to download the file.  If a user selects either "+
                     "the Arabidopsis, Chicken, Human, Mouse, Rat, or Zebrafish annotations "+
                     "from the European Bioinformatics Institute (EBI) "+
                     "then the cross-reference field will default to the cross-reference file corresponding to that "+
                     "annotation file.";

	 Util.renderDialog(this,szMessage,50,100);
      }
      else if (esource == executeHButton)
      {
	 szMessage = "Pressing the 'Load Saved Settings...' button opens a window to specify a file with "+
                     "saved settings. These files are automatically produced through clicking the disk icon "+
                     "on the main profile interface\n"+
                     "Pressing the execute button causes any of the files that are checked next to "+
                     "'Download the latest' to be "+
                     "downloaded.   If the data file has two or more time points " +
                      "then the clustering and gene enrichment algorithms to execute.  "+
                     "When the "+
                     "algorithms complete a new interface will appear showing all the model temporal profiles "+
                     "and will allow a user to obtain more information about each profile."+
                     "  If the data file has zero or one time points then a simple GO enrichment table "+
                     "will be generated, where the set of genes of interest are those genes in the data file "+
                     "(after filtering in the case of one time point) "+
	     "and the base set are those genes in the data file unioned with those in the pre-filtered gene file.";

	 Util.renderDialog(this,szMessage,50,100);
      }
    }

    /////////////////////////////////////////////////////////////////////////////////
    /**
     * Manages the rendering of the set of option tabbed panels
     */
    public void makeOptionsDialog()
    {
       theOptions = new JDialog(this, "Advanced Options", true);

       Container theDialogContainer = theOptions.getContentPane();

       theDialogContainer.setBackground(Color.white);
       JTabbedPane tabbedPane = new JTabbedPane();
       JComponent panelFiltering = makeFilterPanel();
       panelFiltering.setBackground(lightBlue);

       tabbedPane.addTab("Filtering",null,panelFiltering,"Options pertaining to gene filtering");
       
       JComponent panelModel = makeModelPanel();
       panelModel.setBackground(lightBlue);
       tabbedPane.addTab("Model Profiles",null,
                    panelModel,"Additional options pertaining to model profiles");
       JComponent panelClustering = makeClusteringPanel();
       panelClustering.setBackground(lightBlue);
       tabbedPane.addTab("Clustering Profiles",null,
                   panelClustering,"Options pertaining to clustering model profiles");
      

       JComponent panelAnnotation = makeAnnotationPanel();
       panelAnnotation.setBackground(lightBlue);
       tabbedPane.addTab("Gene Annotations",null,panelAnnotation,
                   "Options pertaining to gene annotations");


       JComponent panelGOAnalysis = makeGOAnalysisPanel();
       panelGOAnalysis.setBackground(lightBlue);
       tabbedPane.addTab("GO Analysis",null,panelGOAnalysis,
                   "Options pertaining to Gene Ontology enrichment analysis");

       theDialogContainer.add(tabbedPane);
       theOptions.pack();

       theOptions.setDefaultCloseOperation(DO_NOTHING_ON_CLOSE);
       theOptions.addWindowListener(new WindowAdapter() {
          public void windowClosing(WindowEvent we) {
		  
             boolean bok = true;
             try
	     {
		 thespinnermingo.commitEdit();
		 thespinnermingolevel.commitEdit();
		 thespinnermaxmissing.commitEdit();
		 thespinnerfilterthreshold.commitEdit();
		 thespinnerexpress.commitEdit();
		 thespinnercorrmodel.commitEdit();
		 thespinneralpha.commitEdit();
		 thespinnerlb.commitEdit();
		 thespinnerpercentile.commitEdit();
		 thespinnersamplegene.commitEdit();
		 thespinnersamplemodel.commitEdit();
		 thespinnersamplepval.commitEdit();                        
	     }
             catch (ParseException ex)
	     {
                Toolkit.getDefaultToolkit().beep(); 
		bok = false; 
             }
                    
             if (bok)
	     {
                theOptions.setVisible(false);
	     }
	  }
       });
    }

    /////////////////////////////////////////////////////////////
    /**
     * Makes sure all the settings displayed on the interface correspond
     * to the variables internal values.
     */
    public void updateSettings()
    {

	theRepeatList.updateSettings(vRepeatFilesDEF,balltimeDEF);

        if (brandomgoDEF)
	{
           randomgoButton.setSelected(true);
	}
	else 
	{
           bfgoButton.setSelected(true);
	}

        if (nfdrDEF == 2)
	{
           bfButton.setSelected(true);
	}
	else if (nfdrDEF == 1)
	{
           fdrButton.setSelected(true);
	}
	else
	{
           noneButton.setSelected(true);
	}


        if (bmaxminDEF)
	{
           maxminButton.setSelected(true);
	}
	else 
	{
           absButton.setSelected(true);
	}
	permutecheck.setSelected(ballpermuteDEF);
	pcheck.setSelected(bpontoDEF);
	fcheck.setSelected(bfontoDEF);
	ccheck.setSelected(bcontoDEF);
	spotcheck.setSelected(bspotcheckDEF);
	categoryIDField.setText(szcategoryIDDEF);
	orig1Field.setText(szDataFileDEF);
	xrefField.setText(szCrossRefFileDEF);
	chromField.setText(szChromFileDEF);
	goField.setText(szGeneAnnotationFileDEF);
	extraField.setText(szPrefilteredDEF);
	taxonField.setText(sztaxonDEF);
	evidenceField.setText(szevidenceDEF);

	if (nCLUSTERINGMETHODDEF == 0) 
	{
	   thespinnerProfile.setValue(new Integer(nMaxProfilesDEF));
	   thespinnerChange.setValue(new Integer(nMaxUnitDEF));
	}
	else
	{
	   thespinnerProfile.setValue(new Integer(nKDEF));
	   thespinnerChange.setValue(new Integer(nREPDEF));
	}
	thespinnermingo.setValue(new Integer(nMinGoGenesDEF));
	thespinnermingolevel.setValue(new Integer(nMinGOLevelDEF));
	thespinnermaxmissing.setValue(new Integer(nMaxMissingDEF));
	thespinnerfilterthreshold.setValue(new Double(dMinCorrelationRepeatsDEF));
	thespinnerexpress.setValue(new Double(dMinExpressionDEF));
	thespinnercorrmodel.setValue(new Double(dMaxCorrelationModelDEF));
	thespinneralpha.setValue(new Double(dSignificanceLevelDEF));
	thespinnerlb.setValue(new Double(dMinimumCorrelationClusteringDEF));
	thespinnerpercentile.setValue(new Double(dMinimumPercentileClusteringDEF));
	thespinnersamplegene.setValue(new Integer(nNumPermsGeneDEF));
	thespinnersamplemodel.setValue(new Long(nMaxCandidateModelDEF));
	thespinnersamplepval.setValue(new Integer(nSamplesMultipleDEF));
	clusteringmethodcb.setSelectedIndex(nCLUSTERINGMETHODDEF);
	if (nnormalizeDEF == 0)
	{
	   lognormButton.setSelected(true);
	}
	else if (nnormalizeDEF == 1)
	{
	   normButton.setSelected(true);
        }
	else 
	{
	   nonormButton.setSelected(true);
        }

	orgcb.setSelectedIndex(ndbDEF);
	chromcb.setSelectedIndex(nlocationDEF);
	xrefcb.setSelectedIndex(nxrefDEF);

    }

    ///////////////////////////////////////////////
    /**
     * Renders the option panel about gene annotations
     */
    protected JComponent makeAnnotationPanel() 
    {
	evidenceHButton.addActionListener(this);
	taxonHButton.addActionListener(this);
	ontoHButton.addActionListener(this);

        pcheck = new JCheckBox("Biological Process", bpontoDEF);
        fcheck = new JCheckBox("Molecular Function", bfontoDEF);
        ccheck = new JCheckBox("Cellular Component", bcontoDEF);
    
        JPanel ponto = new JPanel();
        JLabel ontoLabel = new JLabel("Only include annotations of type: ", JLabel.TRAILING);
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
        JLabel evidenceLabel = new JLabel("Exclude annotations with these evidence codes:", JLabel.TRAILING);
	evidenceField = new JTextField(szevidenceDEF, JLabel.TRAILING);
        evidenceField.setColumns(20);

	evidenceLabel.setLabelFor(evidenceField);

        JLabel taxonLabel = new JLabel("Only include annotations with these taxon IDs:", JLabel.TRAILING);
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
	int nwidth =  (int) orig1Field.getPreferredSize().getWidth();
	int nheight =   (int) categoryIDButton.getPreferredSize().getHeight();
	taxonField.setPreferredSize(new Dimension(nwidth,nheight));
	evidenceField.setPreferredSize(new Dimension(nwidth,nheight));
	categoryIDField.setPreferredSize(new Dimension(nwidth,nheight));
	taxonField.setMaximumSize(new Dimension(Integer.MAX_VALUE,nheight));
	evidenceField.setMaximumSize(new Dimension(Integer.MAX_VALUE,nheight));
	categoryIDField.setMaximumSize(new Dimension(Integer.MAX_VALUE,nheight));
        SpringUtilities.makeCompactGrid(pevidence,2,3,4,5,6,6);
	categoryIDLabel.setLabelFor(categoryIDField);
        p2.add(categoryIDField);
        p2.add(categoryIDButton);
        p2.add(categoryIDHButton);
        p2.setBackground(lightBlue);
        categoryIDButton.addActionListener(this);
        categoryIDHButton.addActionListener(this);

        JPanel entirepanel = new JPanel();
        BoxLayout layout = new BoxLayout(entirepanel,BoxLayout.Y_AXIS);
        entirepanel.setLayout(layout);
        p2.setBackground(lightBlue);

	entirepanel.add(ponto);
	entirepanel.add(pevidence);

        entirepanel.add(p2);
        entirepanel.add(Box.createRigidArea(new Dimension(0,50)));

        entirepanel.setLayout(layout);
        entirepanel.setBackground(lightBlue);

        return entirepanel;
    }

    ///////////////////////////////////////////////////////////
    /**
     * Render the GO Analysis options panels
     */
    protected JComponent makeGOAnalysisPanel() 
    {
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
        if (brandomgoval)
	{
           randomgoButton.setSelected(true);
	}
	else 
	{
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
        SpringUtilities.makeCompactGrid(p,3,3,5,5,5,5);  

        JPanel entirepanel = new JPanel();
        BoxLayout layout = new BoxLayout(entirepanel,BoxLayout.Y_AXIS);
        entirepanel.setLayout(layout);

        entirepanel.add(p);
	entirepanel.add(correctPanel);
        entirepanel.setLayout(layout);
        entirepanel.setBackground(lightBlue);

        return entirepanel;
    }



    /**
     *Responsible for downloading the file. ntype specifies the type of file we are downloading.
     */
    public boolean getFile(String szURL, String szoutfile,int ntype) 
    {
       final JFrame fframe = this;
       try
       {
	  if (ntype == 2)
	  {
	     BiomartAccess ba = new BiomartAccess("biomart_species.txt");
	     String szspecies = szchromval.substring(0,szchromval.indexOf("."));
	     ba.updateGeneFile(szspecies,szchromval,npercentdone,ntype,lockpd);
	     return true;
	  }
	  else
          {
	     URL theURL = new URL(szURL);
	     URLConnection theurlc = theURL.openConnection();

  	     int ntotal = theurlc.getContentLength();

  	     InputStream stream = theurlc.getInputStream();
             BufferedInputStream in = new BufferedInputStream(stream);
             FileOutputStream file = new FileOutputStream(szoutfile);
             BufferedOutputStream out = new BufferedOutputStream(file);
             int ni;
             int nread =0;
	     npercentdone[ntype] =0;
	     byte[] b = new byte[1024];
	     int nlastpercentdone = 0;
             while ((ni = in.read(b)) != -1) 
             {
  	        nread += ni;
	        double dpercent = ((double) nread)/ntotal;

	        synchronized (lockpd)
	        {
	           npercentdone[ntype] = (int)(100*dpercent); 
		   if (nlastpercentdone != npercentdone[ntype])
		   {
		      nlastpercentdone = npercentdone[ntype];
  		      lockpd.notifyAll();
		   }
		} 
	        out.write(b,0,ni);
	     }
	     out.flush();
  	     stream.close();
	     in.close();
  	     file.close();
             out.close();
             return true;
	  }
       }
       catch (Exception ex)
       {
          final Exception fex = ex;

          synchronized (lockpd)
          {
	     nexceptions++;
	     bexception[ntype] = true;
	     bdownloading[ntype] = false;
             lockpd.notifyAll();
             if (nexceptions == 1)
             {
                javax.swing.SwingUtilities.invokeLater(new Runnable() 
                {
                   public void run() 
                   {
                      JOptionPane.showMessageDialog(fframe, fex.toString(), 
                                "Exception thrown", JOptionPane.ERROR_MESSAGE);

	           }
	        });
	     }
	  }
	  return false;  
       }
    }

    ////////////////////////////////////////////////////////////////////
    /**
     * Makes panel of options for the clustering model profiles
     */
    protected JComponent makeClusteringPanel()
    {
        JPanel p = new JPanel(new SpringLayout());
        p.add(lbLabel);
        p.add(thespinnerlb);
        p.add(lbHButton);     
        lbHButton.addActionListener(this);
        p.add(percentileLabel);
        p.add(thespinnerpercentile);
        p.add(percentileHButton); 
        percentileHButton.addActionListener(this);
        p.setBackground(lightBlue);
        SpringUtilities.makeCompactGrid(p,2,3,5,5,5,5);    

        JPanel entirepanel = new JPanel();
        BoxLayout layout = new BoxLayout(entirepanel,BoxLayout.Y_AXIS);
        entirepanel.add(p);
        entirepanel.setLayout(layout);
        entirepanel.setBackground(lightBlue);

        return entirepanel;
    }

    ///////////////////////////////////////////////////////////////////////////
    /**
     * Makes panel of options related to the selection and significance of model profiles
     */
    protected JComponent makeModelPanel()
    {

	JPanel p = new JPanel(new SpringLayout());

        p.add(corrmodelLabel);
        p.add(thespinnercorrmodel);
        p.add(corrmodelHButton);
        corrmodelHButton.addActionListener(this);

        p.add(samplemodelLabel); 
        p.add(thespinnersamplemodel);
        p.add(samplemodelHButton);    
        samplemodelHButton.addActionListener(this);

        p.add(samplegeneLabel);
        p.add(thespinnersamplegene);
        p.add(samplegeneHButton);
        samplegeneHButton.addActionListener(this);

        p.add(alphaLabel);             
        p.add(thespinneralpha);
        p.add(alphamodelHButton);
        alphamodelHButton.addActionListener(this);
   
        SpringUtilities.makeCompactGrid(p,4,3,5,5,5,5); 

        //JPanel correctPanel = new JPanel();
        bfButton = new JRadioButton("Bonferroni");
        bfButton.setBackground(lightBlue);
        fdrButton = new JRadioButton("False Discovery Rate");
        fdrButton.setBackground(lightBlue);
        noneButton = new JRadioButton("None");
        noneButton.setBackground(lightBlue);
        ButtonGroup group = new ButtonGroup();
        if (nfdrDEF == 2)
	{
           bfButton.setSelected(true);
	}
	else if (nfdrDEF == 1)
	{
           fdrButton.setSelected(true);
	}
	else
	{
           noneButton.setSelected(true);
	}
        group.add(bfButton);
        group.add(fdrButton);
        group.add(noneButton);
        JPanel correctpanel = new JPanel();
	correctpanel.add(new JLabel("Correction Method:")); 
        correctpanel.add(bfButton); 
        correctpanel.add(fdrButton);  
	correctpanel.add(noneButton);
        correctpanel.setBackground(lightBlue);
	correctpanel.add(methodHButton);
        methodHButton.addActionListener(this);


	JPanel ppermute = new JPanel();
	ppermute.add(permutecheck);
	permutecheck.setBackground(lightBlue);
        ppermute.setBackground(lightBlue);


	if (ballpermuteval)
	{
	    permutecheck.setSelected(true);
	}
	else
	{
	    permutecheck.setSelected(false);
	}

	ppermute.add(permuteHButton);
        permuteHButton.addActionListener(this);

        JPanel entirepanel = new JPanel();
        BoxLayout layout = new BoxLayout(entirepanel,BoxLayout.Y_AXIS);
        entirepanel.setLayout(layout);
        p.setBackground(lightBlue);
        entirepanel.add(p);
	entirepanel.add(ppermute);
        entirepanel.add(correctpanel);
        entirepanel.setBackground(lightBlue);
        return entirepanel;
    }

    /**
     * Makes panel of gene filtering options
     */
    protected JComponent makeFilterPanel()
    {
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
	filterchoiceLabel = new JLabel(szfilterchoice,JLabel.TRAILING);
        p15.add(filterchoiceLabel);
        maxminButton = new JRadioButton("Maximum\u2212Minimum");
        maxminButton.setBackground(lightBlue);
        absButton = new JRadioButton("Difference from 0");
        absButton.setBackground(lightBlue);

        ButtonGroup group = new ButtonGroup();
        if (bmaxminval)
	{
           maxminButton.setSelected(true);
	}
	else 
	{
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

        SpringUtilities.makeCompactGrid(p,3,3,4,4,4,4);  

        JPanel p2 = new JPanel();
        p2.add(extraLabel);
        p2.add(extraField);

	extraField.setPreferredSize(new Dimension((int) orig1Field.getPreferredSize().getWidth(),
				        (int) extraButton.getPreferredSize().getHeight()));

        p2.add(extraButton);
        p2.add(extraHButton);
        p2.setBackground(lightBlue);
        extraButton.addActionListener(this);
        extraHButton.addActionListener(this);

        JPanel entirepanel = new JPanel();
        BoxLayout layout = new BoxLayout(entirepanel,BoxLayout.Y_AXIS);
        entirepanel.setLayout(layout);
        p.setBackground(lightBlue);
        entirepanel.add(p);
	entirepanel.add(p15);
        entirepanel.add(p2);
        entirepanel.setBackground(lightBlue);
        entirepanel.add(Box.createRigidArea(new Dimension(0,30)));

        return entirepanel;       
    }


    /**
     * Create the GUI and show it.  For thread safety,
     * this method should be invoked from the
     * event-dispatching thread.
     */
    private static void createAndShowGUI() throws FileNotFoundException, IOException
    {
        //Create and set up the window.
 
        JFrame frame = new ST();
        frame.setLocation(10,25);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        //Display the window.
        frame.pack();
        frame.setVisible(true);
    }


    /**
     * The main method which launches STEM
     */
    public static void main(String[] args) throws Exception
    {

        boolean bshowusage = true;
	boolean bbatchmode = false;
	String szBatchInputDir = "";
	String szBatchOutputDir = "";

	if (args.length == 0)
	{
	    bshowusage = false;
	}
        else if (args.length == 2)
	{
           if (args[0].equals("-d"))
	   {
	      szDefaultFile = args[1];
              bshowusage = false;
	   }
	   else if (args[0].equals("-o"))
	   {
	       szBatchGOoutput = args[1];
	       bshowusage = false;
	   }
	}
	else if (args.length == 3)
	{
           if (args[0].equals("-b")) 
	   {
	      szBatchInputDir = args[1];
	      szBatchOutputDir = args[2];
	      bshowusage = false;
	      bbatchmode = true;
	   }
	}
	else if (args.length == 4)
	{
	   if ((args[0].equals("-d"))&&(args[2].equals("-o")))
	   {
	      szDefaultFile = args[1];
	      szBatchGOoutput = args[3];
              bshowusage = false;
	   }
	   else if ((args[0].equals("-o"))&&(args[2].equals("-d")))
	   {
	      szDefaultFile = args[3];
	      szBatchGOoutput = args[1];
	      bshowusage = false;
	   }
	}
	if (bshowusage)
	{
	    System.out.println("USAGE: java ST [[-d defaultfilename.txt][-o GOoutfile.txt]|-b batchInput batchOutputDir]");
            return;
	}
        
	if (bbatchmode)
        {
	    new ST(szBatchInputDir,szBatchOutputDir);
	}
	else
	{
           javax.swing.SwingUtilities.invokeLater(new Runnable() {
		   public void run(){
                     try
                     {
                         createAndShowGUI();
		     }
		     catch(FileNotFoundException ex)
		     {
			 ex.printStackTrace(System.out);
		     }
		     catch(IOException ex)
		     {
			 ex.printStackTrace(System.out);
		     }
               }
           });
	}
    }

}

