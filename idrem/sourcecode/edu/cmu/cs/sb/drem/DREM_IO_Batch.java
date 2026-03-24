package edu.cmu.cs.sb.drem;

import edu.cmu.cs.sb.core.*;
import static edu.cmu.cs.sb.drem.DREM_IO.bProteAll;
import static edu.cmu.cs.sb.drem.DREM_IO.brealXaxisDEF;
import static edu.cmu.cs.sb.drem.DREM_IO.djMethyFile;

import java.io.*;
import java.util.*;

import java.awt.image.*;
import java.text.*;
import java.lang.reflect.*;
import java.net.*;

/**
 * Class implementing the main input interface
 */
public class DREM_IO_Batch {
	String szorganismsourceval;
	String szxrefsourceval;
	boolean bbatchmode;
	boolean btraintest;

	int nexceptions;
	boolean bdisplaycurrent = false;
	String szDefaultFile = "";

	// Main
	boolean ballowmergeDEF = false;
	String szStaticFileDEF = "";
	String szDataFileDEF = "";
	String szInitFileDEF = "";
	boolean bspotcheckDEF = false;
	int nnormalizeDEF = 1;
	String szGeneAnnotationFileDEF = "";
	String szCrossRefFileDEF = "";
	int ndbDEF = 0;
	int nxrefDEF = 0;
	int nstaticsourceDEF = 0;
	int numchildDEF = 3;

	// Repeat
	Vector vRepeatFilesDEF = new Vector();
	boolean balltimeDEF = true;

	// Search Options
	double dCONVERGENCEDEF = 0.01;
	double dMinScoreDEF = 0.0;
	double dDELAYPATHDEF = .15;
	double dDMERGEPATHDEF = .15;
	double dPRUNEPATHDEF = .15;
	double dMINSTDDEVALDEF = 0.0;

	double dMinScoreDIFFDEF = 0;
	double dDELAYPATHDIFFDEF = 0;
	double dDMERGEPATHDIFFDEF = 0;
	double dPRUNEPATHDIFFDEF = 0;
	int ninitsearchDEF = 0;
	int nSEEDDEF = 1260;
	double dNODEPENALTYDEF = 40;
	boolean bPENALIZEDDEF = true;
	boolean bstaticsearchDEF = true;

	// DREM with miRNA data
	String miRNAInteractionDataFile = "";
	String miRNAExpressionDataFile = "";
	boolean checkStatusTF = false;
	boolean checkStatusmiRNA = true;
	boolean miRNATakeLog = true;
	boolean miRNAAddZero = false;
	double miRNAWeight = 1.0;
	double tfWeight = 0.5;
	Vector<String> miRNARepeatFilesDEF = new Vector<String>();
	boolean miRNAalltimeDEF = true;
	boolean filtermiRNAExp = false;

	// Filtering
	String szPrefilteredDEF = "";
	int nMaxMissingDEF = 0;
	double dMinExpressionDEF = 1;
	double dMinCorrelationRepeatsDEF = 0;
	boolean bfilterstaticDEF = false;

	// Gene Annotations
	String sztaxonDEF = "";
	String szevidenceDEF = "";
	boolean bpontoDEF = true;
	boolean bcontoDEF = true;
	boolean bfontoDEF = true;
	String szcategoryIDDEF = "";

	// GO Analysis
	int nSamplesMultipleDEF = 500;
	int nMinGoGenesDEF = 5;
	int nMinGOLevelDEF = 3;
	boolean brandomgoDEF = true;
	boolean bmaxminDEF = false;// true;
	double dpercentDEF = 0;

	// GUI
	boolean brealXaxisDEF = false;
	double dYaxisDEF = 1;
	double dXaxisDEF = 1;
	int nKeyInputTypeDEF = 1;
	double dKeyInputXDEF = 3;
	String SZSTATICDIR = "TFInput";
	String szGeneOntologyFileDEF = "gene_ontology.obo";
	double dnodekDEF = 1;

	// SDREM
	double dProbBindingFunctional = 0.8;
	String regScoreFile = "";
	
	// DECOD
	String fastaFile = "";
	String decodPath = "";
	
        //PROTEOMICS
        String djPPIFile="";
        String djProteFile="";
        Vector<String> ProteRepeatFilesDEF = new Vector<String>();
        boolean ProtealltimeDEF = true;
        Boolean bProteAll=false;
        Boolean bProteTF=false;
        Boolean proteTakeLog=true;
        Boolean proteAddZero=false;
        
        
        //METHYLATION
        String djMethyFile="";
        String djMethyGTF="";
        
        String pweight="1";
        
	long s1;

	boolean bendsearch = false;

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
	boolean brandomgoval = true;
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
	boolean bmaxminval;

	boolean balltime = false;
	boolean bspotincluded;
	boolean badd0 = false;
	// boolean bthreewaysplit;

	// Strings for the labels

	String[] staticsourceArray = { "User provided" };

	String szoutmodelfile;

	public DREM_IO_Batch(String szDefaultFile, String szoutmodelfile) {
		this(szDefaultFile, null, szoutmodelfile);
	}

	// sdremBindingFile is set by the DREMInterface class in SDREM
	public DREM_IO_Batch(String szDefaultFile, String sdremBindingFile,
			String szoutmodelfile) {

		File dir = new File(SZSTATICDIR);

		String[] children = dir.list();
		if (children == null) {
			System.out.println("The directory " + SZSTATICDIR
					+ " was not found." + "Directory not found");
		} else {
			staticsourceArray = new String[children.length + 1];
			staticsourceArray[0] = "User Provided";
			for (int i = 0; i < children.length; i++) {
				// Get filename of file or directory
				staticsourceArray[i + 1] = children[i];
			}
		}

		try {
			this.szDefaultFile = szDefaultFile;
			bbatchmode = true;
			s1 = System.currentTimeMillis();
			this.szoutmodelfile = szoutmodelfile;
			parseDefaults();
			szepsilonval = "" + dMinScoreDEF;
			szprunepathval = "" + dPRUNEPATHDEF;
			szdelaypathval = "" + dDELAYPATHDEF;
			szmergepathval = "" + dDMERGEPATHDEF;
			szepsilonvaldiff = "" + dMinScoreDIFFDEF;
			szprunepathvaldiff = "" + dPRUNEPATHDIFFDEF;
			szdelaypathvaldiff = "" + dDELAYPATHDIFFDEF;
			szmergepathvaldiff = "" + dDMERGEPATHDIFFDEF;
			sznumchildval = "" + numchildDEF;
			szseedval = "" + nSEEDDEF;
			bstaticcheckval = bfilterstaticDEF;
			ballowmergeval = ballowmergeDEF;
			ninitsearchval = ninitsearchDEF;
			szinitfileval = szInitFileDEF;
			bstaticsearchval = bstaticsearchDEF;
			sznodepenaltyval = "" + dNODEPENALTYDEF;
			bpenalizedmodelval = bPENALIZEDDEF;
			szconvergenceval = "" + dCONVERGENCEDEF;
			szminstddeval = "" + dMINSTDDEVALDEF;
			// System.out.println("*"+szStaticFileDEF);
                        
                        //FIXME: HEADLESS MODE
                         System.setProperty("java.awt.headless", "true");
                        
			// If a binding file is specified by SDREM, use that instead
			// of what was read from the defaults file
			if (sdremBindingFile != null) {
				szStaticFileDEF = sdremBindingFile;
			}

			clusterscript(szStaticFileDEF, szCrossRefFileDEF, szDataFileDEF,
					szGeneAnnotationFileDEF, szGeneOntologyFileDEF, ""
							+ nMaxMissingDEF, "" + dMinExpressionDEF, ""
							+ dMinCorrelationRepeatsDEF, ""
							+ nSamplesMultipleDEF, "" + nMinGoGenesDEF, ""
							+ nMinGOLevelDEF, szPrefilteredDEF, balltimeDEF,
					vRepeatFilesDEF, (nnormalizeDEF == 0), false, false,
					bspotcheckDEF, (nnormalizeDEF == 2), szcategoryIDDEF,
					szInitFileDEF, szevidenceDEF, sztaxonDEF, bpontoDEF,
					bcontoDEF, bfontoDEF, brandomgoDEF, bmaxminDEF);

		} catch (Exception ex) {
			ex.printStackTrace(System.out);
		}

	}

	/**
	 * Assigns the initial settings of the parameters based on the contents of
	 * szDefaultFile
	 */
	public void parseDefaults() throws FileNotFoundException,
			IOException {
		// System.out.println("in parse\t"+szDefaultFile);
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
							/*Boolean bProteAll=false;
							 * try { ndbDEF = Integer.parseInt(szvalue); if
							 * ((ndbDEF < 0)|| (ndbDEF >= organisms.length)) {
							 * ndbDEF = 0; } } catch(NumberFormatException ex) {
							 * boolean bfound = false; int nsource = 0; while
							 * ((nsource < organisms.length)&&(!bfound)) { if
							 * (organisms[nsource].equalsIgnoreCase(szvalue)) {
							 * bfound = true; ndbDEF = nsource; } else {
							 * nsource++; } }
							 * 
							 * if (!bfound) { szError += "Warning: "+szvalue
							 * +" is an unrecognized "+
							 * "type for Gene Annotation Source\n"; } }
							 */
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
							/*
							 * int numitems = defaultxrefs.length; try {
							 * nxrefDEF = Integer.parseInt(szvalue); if
							 * ((nxrefDEF < 0)|| (nxrefDEF >= numitems)) {
							 * nxrefDEF = 0; } } catch(NumberFormatException ex)
							 * { boolean bfouparseDefaultsnd = false; int nsource = 0; while
							 * ((nsource < numitems)&&(!bfound)) { if (((String)
							 * defaultxrefs[nsource]).equalsIgnoreCase(szvalue))
							 * { bfound = true; nxrefDEF = nsource; } else {
							 * nsource++; } }
							 * 
							 * if (!bfound) { szError += "Warning: "+szvalue
							 * +" is an unrecognized "+
							 * "type for a Cross_Reference_Source"; } }
							 */
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
							vRepeatFilesDEF = new Vector();
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
						} else if (sztype.equalsIgnoreCase("miRNA-gene_Interaction_Source")) {
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
                                                 // parseDefault for proteomics panel 
                                                } else if (sztype.equalsIgnoreCase("Proteomics_File")){
                                                        djProteFile=szvalue;
                                                // parseDefault for relative proteomics weight
                                                } else if (sztype.equalsIgnoreCase("Proteomics_Relative_Weight")){
                                                        pweight=szvalue;
                                                } else if ((sztype
								.equalsIgnoreCase("Repeat_Prote_Data_Files(comma delimited list)"))
								|| (sztype
										.equalsIgnoreCase("Repeat_Prote_Data_Files"))) {
							ProteRepeatFilesDEF = new Vector();
							StringTokenizer stRepeatList = new StringTokenizer(
									szvalue, ",");
							while (stRepeatList.hasMoreTokens()) {
								ProteRepeatFilesDEF.add(stRepeatList.nextToken());
							}
                                                } else if ((sztype.equalsIgnoreCase("Repeat_Prote_Data_is_from"))
								|| (sztype
										.equalsIgnoreCase("Repeat_Prote_Data_is_from[Different time periods,The same time period]"))) {
							if (szvalue.equalsIgnoreCase("Different time periods")) {
								ProtealltimeDEF = true;
							} else if (szvalue.equalsIgnoreCase("The same time period")) {
								ProtealltimeDEF = false;
							} else if (!szvalue.equals("")) {
								szError += "WARNING: '"
										+ szvalue
										+ "' is an invalid value for "
										+ "Repeat Prote Data is from it must be either "
										+ "'Different time periods' or 'The same time period'\n";
							}
                                                } else if (sztype.equalsIgnoreCase("Normalize_Prote_Data[Log normalize data,Normalize data,No normalization/add 0]")) {
							if (szvalue.equalsIgnoreCase("Log normalize data")) {
								proteTakeLog = true;
								proteAddZero = false;
							} else if (szvalue.equalsIgnoreCase("Normalize data")) {
								proteTakeLog = false;
								proteAddZero = false;
							} else if (szvalue.equalsIgnoreCase("No normalization/add 0")) {
								proteTakeLog = false;
								proteAddZero = true;
							} else {
								throw new IllegalArgumentException(
										szvalue+ " is an invalid argument for Normalize_Prote_Data");
							}
                                                } else if (sztype.equalsIgnoreCase("Use Proteomics[No,TF,All]")){
                                                        if (szvalue.equalsIgnoreCase("No")){
                                                            bProteTF=false;
                                                            bProteAll=false;
                                                        } else if (szvalue.equalsIgnoreCase("TF")){
                                                            bProteTF=true;
                                                            bProteAll=false;
                                                        } else if (szvalue.equalsIgnoreCase("All")){
                                                            bProteAll=true;
                                                            bProteTF=false;
                                                        } else{
                                                               throw new IllegalArgumentException(
                                                                       szvalue+" is not valid"
                                                               );
                                                        }
                                                    
                                                } else if (sztype.equalsIgnoreCase("PPI File")){
                                                        djPPIFile=szvalue;
                                             
                                                } else if (sztype.equalsIgnoreCase("Epigenomic_File")){
                                                        // parseDefault for epigenomics panel
                                                        djMethyFile=szvalue;
                                                } else if (sztype.equalsIgnoreCase("GTF File")){
                                                        djMethyGTF=szvalue;
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
	public void errorcheck(String[] origgenes, String[] repeatgenes,
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
	synchronized public DREM_DataSet buildset(String szorganismsourceval,
			String szxrefsourceval, String szxrefval, String szexp1val,
			String szgoval, String szgocategoryval, int nmaxmissing,
			double dexpressedval, double dmincorrelation, int nsamplespval,
			int nmingo, int nmingolevel, String szextraval, boolean balltime,
			Vector repeatnames, boolean btakelog, boolean bspotincluded,
			boolean badd0, String szcategoryIDval, String szevidenceval,
			String sztaxonval, boolean bpontoval, boolean bcontoval,
			boolean bfontoval, boolean brandomgoval, boolean bmaxminval)
			throws Exception {
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
					theDataSet1fm = new DREM_DataSet(
							theDataSet1.filterMissing1point(), theDataSet1.tga);
					theDataSet1fm = new DREM_DataSet(
							theDataSet1fm.filtergenesthreshold1point(),
							theDataSet1fm.tga);
				} else {
					theDataSet1fm = theDataSet1;
				}
				return theDataSet1fm;
			} else {
				String[] origgenes = theDataSet1.genenames;
				theDataSet1 = new DREM_DataSet(theDataSet1.logratio2(),
						theDataSet1.tga);
				theDataSet1 = new DREM_DataSet(
						theDataSet1.averageAndFilterDuplicates(),
						theDataSet1.tga);

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
						theOtherSet = new DREM_DataSet(
								theOtherSet.averageAndFilterDuplicates(),
								theOtherSet.tga);
						// gene values in log ratio before averaging stored

						repeatSets[nset] = theOtherSet;
					}
					theDataSetsMerged = new DREM_DataSet(
							theDataSet1.mergeDataSets(repeatSets),
							theDataSet1.tga);
					theDataSetsMerged = new DREM_DataSet(
							theDataSetsMerged.filterdistprofiles(theDataSet1,
									repeatSets), theDataSetsMerged.tga);
				} else {
					theDataSetsMerged = theDataSet1;
				}

				theDataSetsMerged = new DREM_DataSet(
						theDataSetsMerged.filterMissing(),
						theDataSetsMerged.tga);
				theDataSetsMerged = new DREM_DataSet(
						theDataSetsMerged.filtergenesthreshold2(),
						theDataSetsMerged.tga);

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
					theDataSet1fm = new DREM_DataSet(
							theDataSet1.filterMissing1point(), theDataSet1.tga);
					theDataSet1fm = new DREM_DataSet(
							theDataSet1fm.filtergenesthreshold1point(),
							theDataSet1fm.tga);
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
					theDataSetsMerged = new DREM_DataSet(
							theDataSet1.mergeDataSets(repeatSets),
							theDataSet1.tga);
				} else {
					theDataSetsMerged = theDataSet1;
				}

				theDataSetsMerged = new DREM_DataSet(
						theDataSetsMerged.logratio2(), theDataSetsMerged.tga);
				theDataSetsMerged = new DREM_DataSet(
						theDataSetsMerged.averageAndFilterDuplicates(),
						theDataSetsMerged.tga);
				// gene values before averaging stored
				theDataSetsMerged = new DREM_DataSet(
						theDataSetsMerged.filterMissing(),
						theDataSetsMerged.tga);
				theDataSetsMerged = new DREM_DataSet(
						theDataSetsMerged.filtergenesthreshold2(),
						theDataSetsMerged.tga);

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

	// ///////////////////////////////////////////////////////////////////////
	/**
	 * A control method that handles the response for when the execute button on
	 * the interface is pressed including building the data set, running the
	 * DREM modeling procedure, and displaying the results
	 */
	public void clusterscript(
			String szstaticFileval, // szstaticFieldval -- before reference
									// class variable
			String szxrefval, String szexp1val, String szgoval,
			String szgocategoryval, String szmaxmissingval,
			String szexpressedval, String szfilterthresholdval,
			String szsamplepval, String szmingoval, String szmingolevelval,
			String szextraval, boolean balltime, Vector repeatnames,
			boolean btakelog, boolean bgetxref, boolean bgetgoann,
			boolean bspotincluded, boolean badd0, String szcategoryIDval,
			String szinitfileval, String szevidenceval, String sztaxonval,
			boolean bpontoval, boolean bcontoval, boolean bfontoval,
			boolean brandomgoval, boolean bmaxminval) throws Exception {

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

			bendsearch = false;

			final DREM_DataSet thefDataSetfmnel = DREM_IO.buildset(
					szorganismsourceval, szxrefsourceval, szxrefval, szexp1val,
					szgoval, szgocategoryval, nmaxmissing, dexpressedval,
					dmincorrelation, nsamplespval, nmingo, nmingolevel,
					szextraval, balltime, repeatnames, btakelog, bspotincluded,
					badd0, szcategoryIDval, szevidenceval, sztaxonval,
					bpontoval, bcontoval, bfontoval, brandomgoval, bmaxminval);

			DataSetCore theMIRNADataSet = null;
			if (checkStatusmiRNA && miRNAExpressionDataFile != null
					&& !miRNAExpressionDataFile.equals("")) {
				theMIRNADataSet = DREM_IO.buildMIRNAset(null, null, null,
						miRNAExpressionDataFile, "", null, nmaxmissing,
						dexpressedval, dmincorrelation, nsamplespval, nmingo,
						nmingolevel, szextraval, miRNAalltimeDEF,
						miRNARepeatFilesDEF, miRNATakeLog, false, miRNAAddZero,
						szcategoryIDval, szevidenceval, sztaxonval, bpontoval,
						bcontoval, bfontoval, brandomgoval, bmaxminval);
			}
			
                        
                        DataSetCore theProteDataSet=null; //FIXME update proteDataset Batch mode here;
                         
                        if((bProteAll||bProteTF) && !(djProteFile.equals(""))){ 
                            theProteDataSet=DREM_IO.buildProteset(null,null,null,
                                    djProteFile,"",null,nmaxmissing,
                                    dexpressedval,dmincorrelation,nsamplespval,nmingo,
                                    nmingolevel,szextraval,ProtealltimeDEF,
                                    ProteRepeatFilesDEF,proteTakeLog,false,proteAddZero,
                                    szcategoryIDval,szevidenceval,sztaxonval,bpontoval,
                                    bcontoval,bfontoval,brandomgoval,bmaxminval);
                         }
                         
            
                        String miRNAExFile=miRNAExpressionDataFile;
                        String szExFile=szexp1val;
                   
                        HashMap<String,HashMap<String,Double>> methyGeneScoreMap=null;
                        
                         if (!(djMethyFile.equals("")) && !(djMethyGTF.equals(""))){
                            methyGeneScoreMap=getMethyScore(djMethyFile,djMethyGTF);
                        }
			// --------------------------------------------------------------------------------------------

			DREM_Timeiohmm thetimehmm = null;

			thetimehmm = new DREM_Timeiohmm(thefDataSetfmnel, szstaticFileval,
					sznumchildval, szepsilonval, szprunepathval,
					szdelaypathval, szmergepathval, szepsilonvaldiff,
					szprunepathvaldiff, szdelaypathvaldiff, szmergepathvaldiff,
					szseedval, bstaticcheckval, ballowmergeval, ninitsearchval,
					szinitfileval, null, null, null, null, null, null,
					bstaticsearchval, brealXaxisDEF, dYaxisDEF, dXaxisDEF,
					dnodekDEF, nKeyInputTypeDEF, dKeyInputXDEF, dpercentDEF,
					null, "", sznodepenaltyval, bpenalizedmodelval,
					szconvergenceval, szminstddeval,
					staticsourceArray[nstaticsourcecb], checkStatusTF,
					checkStatusmiRNA, miRNAInteractionDataFile,
                                        szExFile,
					theMIRNADataSet,miRNAExFile,
                                        theProteDataSet,djProteFile,djPPIFile,bProteAll, //proteomics
                                        methyGeneScoreMap,//methy promoter
                                        fastaFile, decodPath, regScoreFile, dProbBindingFunctional, miRNAWeight, tfWeight,
					filtermiRNAExp,pweight);

			((DREM_GoAnnotations) thetimehmm.theDataSet.tga).buildRecDREM(
					thetimehmm.treeptr, thetimehmm.theDataSet.genenames);
                                        
			thetimehmm.traverse(thetimehmm.treeptr, 0, true);
			thetimehmm.traverse(thetimehmm.treeptr, 0, false);

			final DREM_Timeiohmm fthetimehmm = thetimehmm;

			try {   
                                
                                
                                //export 
                                /*
                                DREMGui theDREMGui = new DREMGui(fthetimehmm,
								fthetimehmm.treeptr, brealXaxisDEF, dYaxisDEF,
								dXaxisDEF, nKeyInputTypeDEF, dKeyInputXDEF,
								dpercentDEF, "(Final Model)", dnodekDEF);
                                */
                                jsonExport jse=new jsonExport();
                                
                                jse.exportJsonFile(fthetimehmm,fthetimehmm.treeptr);
				PrintWriter pw = new PrintWriter(new FileWriter(szoutmodelfile));
				pw.print(fthetimehmm.saveString(fthetimehmm.treeptr));
				pw.close();

				// Only need to save the activity scores when running SDREM
				if(fthetimehmm.regPriorsFile != null && !fthetimehmm.regPriorsFile.equals(""))
				{
					// TODO Temporarily commented out these function calls until
					// the rest of the SDREM code merge is complete.  Need a
					// more elegant way to save activity scores.
					/*
					File outFile = new File(szoutmodelfile);
					DREMGui_SaveModel.saveActivityScoresDynamic(outFile,
							fthetimehmm, fthetimehmm.treeptr);
					DREMGui_SaveModel.saveActivityScores(outFile, fthetimehmm);
					*/
				}
			} catch (IOException ex) {
				ex.printStackTrace(System.out);
			}
			long e1 = System.currentTimeMillis();
			System.out.println("Time: " + (e1 - s1) + "ms");
		}
	}
        
        //
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
        
        

}
