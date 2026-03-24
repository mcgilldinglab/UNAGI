package edu.cmu.cs.sb.drem;

import com.google.gson.Gson;
import edu.cmu.cs.sb.core.*;
import edu.cmu.cs.sb.drem.DREM_Timeiohmm.SigTFRecv2;
import edu.cmu.cs.sb.drem.DREM_Timeiohmm.Treenode;
import java.math.*;

import javax.swing.JFrame;
import javax.swing.ToolTipManager;
import java.awt.geom.*;
import edu.umd.cs.piccolo.PCanvas;
import edu.umd.cs.piccolo.PCamera;
import edu.umd.cs.piccolo.PNode;
import edu.umd.cs.piccolo.PLayer;
import edu.umd.cs.piccolo.event.PDragEventHandler;
import edu.umd.cs.piccolo.event.PBasicInputEventHandler;
import edu.umd.cs.piccolo.event.PInputEvent;
import edu.umd.cs.piccolo.nodes.PPath;
import edu.umd.cs.piccolox.PFrame;
import edu.umd.cs.piccolo.nodes.PText;
import edu.umd.cs.piccolo.nodes.PImage;
import edu.umd.cs.piccolox.nodes.PLine;
import java.util.*;
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.text.*;
import java.io.*;


/**
 * Class for the main interface window of a DREM regulatory map
 */
public class DREMGui extends PFrame implements ComponentListener {

	boolean battachlabels = true;
	static int SCREENWIDTH = 800;
	static int SCREENHEIGHT = 600; //FIXME: CHANGE THE SCREEN HEIGHT
	static int LEFTBUFFER = 25;
	static int RIGHTBUFFER = 75;
	static double TOPBUFFER = 10;
	static double BOTTOMBUFFER = 100;
	double REALHEIGHT = 0; // not constant
	static int TICKLENGTH = 15;
	static double INITKEYLEVEL = .001;
	static int BUTTONRECWIDTH = 110;
	double INITLEFT;
	boolean binit = true;
	static Object datasetlock = new Object();
	static Color buttonColor = new Color(255, 246, 143);
	DREM_Timeiohmm theTimeiohmm;
	double[][] data;
	int[][] pmavalues;
	String[] genenames;
	PCanvas canvas;
	PLine[] plArray;
	PLayer theLayer;
	boolean displayTF = true, displayMIRNA = true;
	boolean colorSigTFs = true;
	Hashtable<PPath, Point2D> lineendpoints = new Hashtable<PPath, Point2D>();

	ArrayList<PNode> hideList = new ArrayList<PNode>();// array list of nodes
	// and edges whose
	// visibility can be toggled
	ArrayList<SigInfoRec> hidesigList;
	ArrayList<PText> hidegolabelsList;
	ArrayList<PText> hidetfsetlabelsList;
	ArrayList<PText> hidepredictlabelsList;
	ArrayList<PText> hidegenesetlabelsList;
	TreeSet<CircleRec> circleSet = new TreeSet<CircleRec>(
			new CircleRecCompare());
	Hashtable<PNode, Color> htColors;
	Hashtable<BigInteger, Integer> htHidden = new Hashtable<BigInteger, Integer>();
	HashSet<CircleID> htRequired = new HashSet<CircleID>();
	Hashtable<PNode, CircleID> htNodes = new Hashtable<PNode, CircleID>();
	Hashtable<BigInteger, ArrayList<Integer>> htPathToLineSet = new Hashtable<BigInteger, ArrayList<Integer>>();
	Hashtable<String, Boolean> htTextVisible;
	JFrame filterStaticFrame;
	JFrame filterGOFrame;
	JFrame defineFrame;
	JFrame predictFrame;
	JFrame keyinputFrame;
	JFrame yscaleFrame;
	JFrame saveDREMFrame;
	JFrame saveModelFrame;
	SelectedNodeRec theSelectedRec = new SelectedNodeRec();
	DREMGui_GOFilter theGOFilter;
	DREMGui_InterfaceOptions theYScalegui;
	DREM_Timeiohmm.Treenode treecopy;

	PNode genetableButton;
	PNode setButton;
	PNode gotableButton;
	PNode staticButton;
	PNode gofilterButton;
	PNode scaleButton;
	PNode predictButton;
	PNode hideButton;
	PNode timeButton;
	PNode siginButton;
	PNode saveDREMButton;
	PNode saveModelButton;
	PText hideText;
	PImage helpButton;
	int nheight;
	int nwidth;
	int ncolortime = 0;

	int ncircleID = 0;
	double dheightunits;
	double dminheightunits;
	double dwidthunits;
	double dnodek = 1;
	double[] dwidthunitsInterval;
	double[] dwidthunitsCum;
	double dmax;
	double dmin;
	double dkeyinputpvalue = INITKEYLEVEL;
	double dkeymiRNAinputpvalue = INITKEYLEVEL;
	double dsplitpercent = 0;
	Color SPLITCOLOR = Color.green;

	boolean bapplygenesetlabels = false;;
	boolean bapplytfsetlabels = false;
	boolean bapplygolabels = false;

	boolean bholdedge = false;
	boolean bfiltergo = false;
	boolean bfiltergeneset = false;
	boolean bfilterinput = false;
	boolean brealXaxis = false;
	PText filterText;

	Color keyInputLabelColor = Color.black;
	Color goLabelColor = Color.black;
	Color predictLabelColor = Color.black;
	Color genesetLabelColor = Color.orange;
	Color tfLabelColor = Color.red;

	boolean binvalidreal = false;// whether sampling rate input is valid

	boolean[] bGOVisible;
	boolean[] bSetVisible;
	boolean[] bPathVisible;
	boolean[] bTFVisible;
	boolean bglobalVisible;

	boolean blowestbelow = true;
	boolean blowestabove = true;
	boolean bmiddlebelow = true;
	boolean bmiddleabove = true;
	boolean bhighestbelow = true;
	boolean bhighestabove = true;
	boolean blowbelow = true;
	boolean blowabove = true;

	boolean bglobalnode = true;
	boolean bshowpredict = false;
	boolean bshowkeyinputs = true;
	boolean bshowgolabels = true;
	boolean bshowtfsetlabels = true;
	boolean bshowgenesetlabels = true;
	int nKeyInputType;
	NumberFormat nf3;

	int numcolor = 0;
	int nparentcolorindex = 0;
	Hashtable<Integer, ArrayList<PPath>> htColorIDtoLinesList = new Hashtable<Integer, ArrayList<PPath>>();
	double dscaley;
	double dscalex = 0.5;

	Hashtable<Integer, Color> htColorIDtoColor = new Hashtable<Integer, Color>();
	Hashtable<String, Integer> htLineIDtoColorID = new Hashtable<String, Integer>();
	Hashtable<Integer, ArrayList<PNode>> htColorIDtoCircleList = new Hashtable<Integer, ArrayList<PNode>>();
	
	HashSet<String> sigRegs = new HashSet<String>();
	
	BigInteger[] storedbestpath;
	boolean bsavedchange;

	double dinitmin;
	double dinitmax;
	Color[] lineColorsA;

	Color edgeColorsTriples[][] = {
			{
					new Color((float) 32 / 255, (float) 178 / 255,
							(float) 170 / 255, (float) 1),
					new Color((float) 255 / 255, (float) 0 / 255,
							(float) 0 / 255, (float) 1),// red
					new Color((float) 255 / 255, (float) 0 / 255,
							(float) 255 / 255, (float) 1) },// pink

			{
					new Color((float) 255 / 255, (float) 128 / 255,
							(float) 0 / 255, (float) 1),// orange
					new Color((float) 128 / 255, (float) 128 / 255,
							(float) 128 / 255, (float) 1),// gray
					new Color((float) 205 / 255, (float) 51 / 255,
							(float) 51 / 255, (float) 1) },// brown3

			{
					new Color((float) 0 / 255, (float) 153 / 255,
							(float) 153 / 255, (float) 1),
					new Color((float) 139 / 255, (float) 115 / 255,
							(float) 85 / 255, (float) 1),// burlywood4
					new Color((float) 204 / 255, (float) 0 / 255,
							(float) 204 / 255, (float) 1) },// beige

			{
					new Color((float) 46 / 255, (float) 139 / 255,
							(float) 87 / 255, (float) 1),// sea green
					new Color((float) 128 / 255, (float) 0 / 255,
							(float) 255 / 255, (float) 1),
					new Color((float) 128 / 255, (float) 0 / 255,
							(float) 128 / 255, (float) 1) },// purple},//light
			// green

			{
					new Color((float) 205 / 255, (float) 91 / 255,
							(float) 69 / 255, (float) 1),// coral
					new Color((float) 14 / 255, (float) 59 / 255,
							(float) 59 / 255, (float) 1),// rosy brown3
					new Color((float) 255 / 255, (float) 128 / 255,
							(float) 255 / 255, (float) 1) },

			{
					new Color((float) 85 / 255, (float) 107 / 255,
							(float) 47 / 255, (float) 1),// dark olive green
					new Color((float) 128 / 255, (float) 128 / 255,
							(float) 255 / 255, (float) 1),// blue
					new Color((float) 255 / 255, (float) 0 / 255,
							(float) 128 / 255, (float) 1) },

			{
					new Color((float) 162 / 255, (float) 162 / 255,
							(float) 104 / 255, (float) 1),// gray
					new Color((float) 0 / 255, (float) 102 / 255,
							(float) 102 / 255, (float) 1),
					new Color((float) 255 / 255, (float) 102 / 255,
							(float) 102 / 255, (float) 1) },// pink

			{
					new Color((float) 153 / 255, (float) 153 / 255,
							(float) 255 / 255, (float) 1),
					new Color((float) 153 / 255, (float) 102 / 255,
							(float) 0 / 255, (float) 1),
					new Color((float) 0 / 255, (float) 102 / 255,
							(float) 102 / 255, (float) 1) },

			{
					new Color((float) 255 / 255, (float) 0 / 255,
							(float) 102 / 255, (float) 1),
					new Color((float) 51 / 255, (float) 51 / 255,
							(float) 51 / 255, (float) 1),
					new Color((float) 102 / 255, (float) 255 / 255,
							(float) 102 / 255, (float) 1) },

			{
					new Color((float) 0 / 255, (float) 102 / 255,
							(float) 153 / 255, (float) 1),
					new Color((float) 155 / 255, (float) 0 / 255,
							(float) 0 / 255, (float) 1),
					new Color((float) 102 / 255, (float) 0 / 255,
							(float) 102 / 255, (float) 1) },

			{
					new Color((float) 102 / 255, (float) 0 / 255,
							(float) 102 / 255, (float) 1),
					new Color((float) 146 / 255, (float) 92 / 255,
							(float) 62 / 255, (float) 1),
					new Color((float) 255 / 255, (float) 204 / 255,
							(float) 204 / 255, (float) 1) },// pink

			{
					new Color((float) 209 / 255, (float) 133 / 255,
							(float) 0 / 255, (float) 1),// gray
					new Color((float) 73 / 255, (float) 39 / 255,
							(float) 84 / 255, (float) 1),
					new Color((float) 241 / 255, (float) 104 / 255,
							(float) 104 / 255, (float) 1) },// pink

			{
					new Color((float) 170 / 255, (float) 15 / 255,
							(float) 163 / 255, (float) 1),
					new Color((float) 205 / 255, (float) 102 / 255,
							(float) 29 / 255, (float) 1),// chocoloate
					new Color((float) 205 / 255, (float) 92 / 255,
							(float) 92 / 255, (float) 1) } };// ,indian red

	
        /**
         * Class constructor override
         */
        
        public DREMGui(){
            
        }
        /**
	 * Class constructor
	 */
	public DREMGui(DREM_Timeiohmm theTimeiohmm,
			DREM_Timeiohmm.Treenode treecopy, boolean brealXaxisDEF,
			double dYaxisDEF, double dXaxisDEF, int nKeyInputTypeDEF,
			double dKeyInputXDEF, double dpercentDEF, String szFinal,
			double dnodekDEF) {
		super("iDREM - Main Interface " + szFinal);  
		this.treecopy = treecopy;
		storedbestpath = new BigInteger[theTimeiohmm.storedbestpath.length];
		bsavedchange = theTimeiohmm.bsavedchange;
		for (int nindex = 0; nindex < storedbestpath.length; nindex++) {
			storedbestpath[nindex] = theTimeiohmm.storedbestpath[nindex];
		}

		synchronized (datasetlock) {
			if (DREM_Timeiohmm.BDEBUG) {
				System.out.println("made it!!!!!!!!!");
			}
			dscaley = dYaxisDEF;
			dscalex = dXaxisDEF;
			dnodek = dnodekDEF;
			brealXaxis = brealXaxisDEF;
			nKeyInputType = nKeyInputTypeDEF;
			dsplitpercent = dpercentDEF / 100;
			// rounding to the nearest tenth
			dkeyinputpvalue = Math.pow(10,
					-(Math.round(10 * dKeyInputXDEF)) / 10.0); // 2.51 --> 25.1
			// --> 2.5
			dkeymiRNAinputpvalue = Math.pow(10, -(Math
					.round(10 * dKeyInputXDEF)) / 10.0);

			this.theTimeiohmm = theTimeiohmm;
			data = theTimeiohmm.theDataSet.data;
			pmavalues = theTimeiohmm.theDataSet.pmavalues;
			genenames = theTimeiohmm.theDataSet.genenames;
			bPathVisible = new boolean[data.length];
			bTFVisible = new boolean[data.length];
			bGOVisible = new boolean[data.length];
			bSetVisible = new boolean[data.length];

			for (int nrow = 0; nrow < data.length; nrow++) {
				bTFVisible[nrow] = true;
				bPathVisible[nrow] = true;
				bGOVisible[nrow] = true;
				bSetVisible[nrow] = true;
			}
			plArray = new PLine[data.length];
			bglobalVisible = true;

			htColors = new Hashtable<PNode, Color>();

			addComponentListener(this);
			nf3 = NumberFormat.getInstance(Locale.ENGLISH);
			nf3.setMinimumFractionDigits(3);
			nf3.setMaximumFractionDigits(3);

			dmax = Math.abs(data[0][0]);
			dmin = Math.abs(data[0][0]);
			double[] dmaxRow = new double[data.length];
			double[] dminRow = new double[data.length];

			if (DREM_Timeiohmm.BDEBUG) {
				System.out.println(dmax + "\t" + dmin);
			}

			for (int nrow = 0; nrow < data.length; nrow++) {
				dmaxRow[nrow] = 0;
				dminRow[nrow] = 0;
				for (int ncol = 0; ncol < data[0].length; ncol++) {
					if (pmavalues[nrow][ncol] != 0) {
						if (data[nrow][ncol] < dminRow[nrow]) {
							dminRow[nrow] = data[nrow][ncol];
						}

						if (data[nrow][ncol] > dmaxRow[nrow]) {
							dmaxRow[nrow] = data[nrow][ncol];
						}
					}
				}
			}

			Arrays.sort(dminRow);
			Arrays.sort(dmaxRow);

			dmin = dminRow[0];
			dmax = dmaxRow[dmaxRow.length - 1];
			dinitmin = dmin;
			dinitmax = dmax;
			if (DREM_Timeiohmm.BDEBUG) {
				System.out.println(dinitmin + " " + dinitmax);
			}

			// potential synchronization issue with getting height
			nheight = getHeight();
			REALHEIGHT = (nheight - TOPBUFFER - BOTTOMBUFFER);
			dminheightunits = REALHEIGHT / (dmax - dmin);
			treecopy.dsigma = (dmax - dmin) * 0.02;
			binit = true;

			if (DREM_Timeiohmm.BDEBUG) {
				System.out.println("calling notifyall");
			}

			datasetlock.notifyAll();
		}
                System.out.println("exporting json...");
                exportJsonFile(theTimeiohmm,treecopy);
	}
              
        public void treeWalk(DREM_Timeiohmm.Treenode ptr, ArrayList<DREM_Timeiohmm.Treenode> TNs){
           DREM_Timeiohmm.Treenode[] TNList;
           //DREM_Timeiohmm.Treenode tn;
           if(ptr!=null){
                TNs.add(ptr);
                TNList=ptr.nextptr;
                for (DREM_Timeiohmm.Treenode tn : TNList){
                    treeWalk(tn,TNs);
                }
           }
            
        }
        
        public ArrayList<DREM_Timeiohmm.Treenode> treeWalkBF(DREM_Timeiohmm.Treenode ptr){
            ArrayList<DREM_Timeiohmm.Treenode> AT=new ArrayList<>();
            ArrayList<DREM_Timeiohmm.Treenode> Q=new ArrayList<>();
            ArrayList<DREM_Timeiohmm.Treenode> QP=new ArrayList<>();
            
            DREM_Timeiohmm.Treenode [] TList={ptr};
            DREM_Timeiohmm.Treenode [] tmpList;
            DREM_Timeiohmm.Treenode current;
            QP.add(ptr);
            AT.add(ptr);
            ArrayList<Double> ListVal=new ArrayList<>();
            ArrayList<Integer> ListValIndex=new ArrayList<>();
            while(QP.size()>0){
                Q.clear();
                for (DREM_Timeiohmm.Treenode tn: QP){
                    tmpList=tn.nextptr;
                    ListVal.clear();
                    ListValIndex.clear();
                    
                    for (DREM_Timeiohmm.Treenode tnn:tmpList){
                        if (tnn!=null){
                            ListVal.add(tnn.dmean);
                        }
                    }
                    
                    Collections.sort(ListVal,Collections.reverseOrder());
                    
                    for (DREM_Timeiohmm.Treenode tnn:tmpList){
                        if (tnn!=null){
                            ListValIndex.add(ListVal.indexOf(tnn.dmean));
                        }
                    }
                    
                    for (int i: ListValIndex){
                        AT.add(tmpList[i]);
                        Q.add(tmpList[i]);
                    }
                }
                //
                QP.clear();
                for (DREM_Timeiohmm.Treenode tn: Q){
                    QP.add(tn);
                }
            }
            
            return AT;
        }
        
        //FIXME: export json file
        public void exportJsonFile(DREM_Timeiohmm theTimeiohmm,DREM_Timeiohmm.Treenode ptr){
            DREM_Timeiohmm.Treenode fptr = ptr;
            DREM_Timeiohmm.Treenode[] nptrList;
            
            // get node List
            ArrayList<DREM_Timeiohmm.Treenode> TNs=new ArrayList<>();  
            ArrayList<DREM_Timeiohmm.Treenode> TNNs=new ArrayList<>();
            String[][] tabdata;
            TNNs=treeWalkBF(ptr); 
            ArrayList<JsonNode> JsonNodeList=new ArrayList<>();
            
            int nparent;
            int nID;
            
            double nMean;
            double nSigma;
            boolean[] inNode;
            String[] TimePoints=getTimePoints(theTimeiohmm.theDataSet.szInputFile,theTimeiohmm.theDataSet.badd0);
            String nTime;
            ArrayList<String[]> nodeGO;
            
            int ct=0;
            for (DREM_Timeiohmm.Treenode node: TNNs){
                if(node.bInNode!=null&&TNs.contains(node)==false){
                    TNs.add(node);
                }
            }
            
            for (DREM_Timeiohmm.Treenode node: TNs){
                nparent=TNs.indexOf(node.parent);
                ArrayList<Integer> nchildren=new ArrayList<>();
                for (DREM_Timeiohmm.Treenode child:node.nextptr){
                    nchildren.add(TNs.indexOf(child));      
                }
                Collections.sort(nchildren);
                
                if(nparent!=-1){    
                    nID=TNs.indexOf(node);
                    nTime=TimePoints[node.ndepth];
                    nMean=node.dmean;
                    nSigma=node.dsigma; 
                    inNode=node.bInNode;
                    nodeGO=getTgr(theTimeiohmm,inNode);
                    int childIndex=Arrays.asList(node.parent.nextptr).indexOf(node);
                    tabdata=getEdgeData(theTimeiohmm,node.parent,childIndex);
                    JsonNode JN= new JsonNode(nparent,nchildren,nID,nTime,nMean,nSigma,inNode,tabdata,nodeGO);
                    JsonNodeList.add(JN);          
                }else{
                    JsonNode JN=new JsonNode();
                    JN.nodeID=0;
                    JN.nodetime=TimePoints[0];
                    JN.nodeMean=0;
                    JN.nodeSigma=0;
                    JN.parent=nparent;
                    JN.children=nchildren;
                    JsonNodeList.add(JN);
                }
                System.out.println(ct);
                ct+=1;
            }
            
            System.gc();
            int[][] regTarget=theTimeiohmm.bindingData.gene2RegBindingIndex[0];
            ArrayList<String[]> geneAbList=new ArrayList<String[]>();
            ArrayList<String[]> mirAbList=new ArrayList<String[]>();
            ArrayList<String[]> proteAbList=new ArrayList<String[]>();
            geneAbList=getGeneAbVal(theTimeiohmm.theDataSet.szInputFile);
            
            String mirExpressionString,mirExpressionIDString;
            if (theTimeiohmm.miRNADataSet!=null){
                mirAbList=getGeneAbVal(theTimeiohmm.miRNADataSet.szInputFile);
                //mirExpressionString=new Gson().toJson(theTimeiohmm.miRNADataSet.data);
                mirExpressionString=Arrays.deepToString(theTimeiohmm.miRNADataSet.data);
                mirExpressionIDString=new Gson().toJson(theTimeiohmm.miRNADataSet.genenames);
            }else{
                mirAbList=null;
                mirExpressionString=null;
                mirExpressionIDString=null;
            }
            
            if (theTimeiohmm.proteDataSet!=null){
                proteAbList=getGeneAbVal(theTimeiohmm.proteDataSet.szInputFile);
            }else{
                proteAbList=null;
            }
            
            
            //String geneExpressionString=new Gson().toJson(theTimeiohmm.theDataSet.data);
            String geneExpressionString=Arrays.deepToString(theTimeiohmm.theDataSet.data);
            String jsonString = new Gson().toJson(JsonNodeList);
            String silGeneScoreString=new Gson().toJson(theTimeiohmm.methyGeneScore);
            String regTargetString=new Gson().toJson(regTarget);
            String geneNameString=new Gson().toJson(theTimeiohmm.theDataSet.genenames);
            String regNameString=new Gson().toJson(theTimeiohmm.bindingData.regNames);
            String geneAbListString=new Gson().toJson(geneAbList);
            String mirAbListString=new Gson().toJson(mirAbList);
            String proteAbListString=new Gson().toJson(proteAbList);
            
            ArrayList<String> JsonStringList =new ArrayList<String>();
            JsonStringList.add(jsonString);
            JsonStringList.add(silGeneScoreString);
            JsonStringList.add(regTargetString);
            JsonStringList.add(geneNameString);
            JsonStringList.add(regNameString);
            JsonStringList.add(geneExpressionString);
            JsonStringList.add(mirExpressionString);
            JsonStringList.add(mirExpressionIDString);
            JsonStringList.add(geneAbListString);
            JsonStringList.add(mirAbListString);
            JsonStringList.add(proteAbListString);
            
            StringBuilder jsonsb=new StringBuilder();
            jsonsb.append("[");
            
            for (String s: JsonStringList){
                jsonsb.append(s);
                jsonsb.append(",");
            }
            jsonsb.append("]");
            
            String combinedJson=jsonsb.toString();
            /*
            String combinedJson="["+jsonString+','+silGeneScoreString+','+regTargetString+','+geneNameString+','+regNameString+
                    ','+geneExpressionString+','+mirExpressionString+','+mirExpressionIDString+','+geneAbListString+','+mirAbListString+','+proteAbListString+']';
            */
            
            System.out.println("writing Json..");
            try {
                File srcFolder=new File("lib/viz/");
                String inputGeneString=theTimeiohmm.theDataSet.szInputFile;
                String destFolderString=inputGeneString+"_viz/";
                File destFolder=new File (destFolderString);
                copyFolder(srcFolder, destFolder);
                BufferedWriter bw = new BufferedWriter(new FileWriter(destFolderString + "DREM.json"));
                bw.write("data=" + combinedJson);
                bw.close();
                Desktop.getDesktop().open(new File(destFolderString+"/idrem_result.html"));
            } catch (IOException e){
                e.printStackTrace();
            }
            
        }
        
        //get node go 
        
        public ArrayList<String[]> getTgr(DREM_Timeiohmm theTimeiohmm, boolean[] inNode){
            GoAnnotations.GoResults tgr;
            double[] dweightA;
            dweightA = new double[inNode.length];
		for (int nindex = 0; nindex < dweightA.length; nindex++) {
			if (inNode[nindex]) {
				dweightA[nindex] = 1;
			} else {
				dweightA[nindex] = 0;
			}
		}
            tgr = theTimeiohmm.theDataSet.tga.getCategory(theTimeiohmm.theDataSet.genenames, dweightA);
            String goID,goCat,goPvalue,goPvalueCorrected;
            String[] GoItem;
            ArrayList<String[]> nodeGO=new ArrayList<String[]>();
            for (int i=0;i<tgr.goRecArray.length;i++){
                goID=tgr.goRecArray[i].szgoid;
                goCat=tgr.goRecArray[i].szgocategory;
                goPvalue=Double.toString(tgr.goRecArray[i].dpvalue);
                goPvalueCorrected=Double.toString(tgr.goRecArray[i].dcorrectedpvalue);
                GoItem=new String[]{goID,goCat,"p-value: "+goPvalue,"corrected p-value:" +goPvalueCorrected};
                nodeGO.add(GoItem);
            }
            
            return nodeGO;
        }
        
        // get Absolute expression value 
        public ArrayList<String[]> getGeneAbVal(String szinputFile){
           
            ArrayList<String[]> GeneAbValList=new ArrayList<String[]>();
            try{
                BufferedReader br=new BufferedReader(new FileReader(szinputFile));
                String Line;
                String[] LineSplit;
                while((Line=br.readLine())!=null){
                    LineSplit=Line.split("\t");
                    GeneAbValList.add(LineSplit);
                }
            }catch (IOException e){
                e.printStackTrace();
            }
            return GeneAbValList;
        }
        
        // output the edgeTable
        //FIXME: export edge table
        public void exportEdgeTable(DREM_Timeiohmm theTimeiohmm,DREM_Timeiohmm.Treenode ptr){
            DREM_Timeiohmm.Treenode fptr = ptr;
            DREM_Timeiohmm.Treenode[] nptrList;
            
            ArrayList<DREM_Timeiohmm.Treenode> TNs=new ArrayList<>();
            ArrayList<String[][]> DataList=new ArrayList<>();
            ArrayList<Integer[]> DataListIndex=new ArrayList<>();
            
            String[][] tabdata;
            HashMap<Integer,Integer> childMap=new HashMap<>();
            int[] childIterm;
            ArrayList<Integer> ChildList=new ArrayList<>();
            TNs=treeWalkBF(ptr); 
            
            DREM_Timeiohmm.Treenode child;
            DREM_Timeiohmm.Treenode node;
            int childListIndex;
            for (int nodeIndex=0;nodeIndex<TNs.size();nodeIndex++){
                for (int childIndex=0;childIndex<TNs.get(nodeIndex).nextptr.length;childIndex++){
                    node=TNs.get(nodeIndex);
                    child=node.nextptr[childIndex];
                    childListIndex=TNs.indexOf(child);
                    if (child!=null){
                        tabdata=getEdgeData(theTimeiohmm,node,childIndex);
                        DataList.add(tabdata);
                        Integer[] dataIndex=new Integer[3];
                        dataIndex[0]=nodeIndex;
                        dataIndex[1]=childListIndex;
                        dataIndex[2]=child.ndepth;
                        DataListIndex.add(dataIndex);
                    }
                }
            }
            
            // 
            StringBuffer outstr=new StringBuffer();
            String tabline;
            String tab;
            
            String[][] tabd;
            for (int i=0;i<DataList.size();i++){
                tab=">"+DataListIndex.get(i)[0]+','+DataListIndex.get(i)[1]+','+DataListIndex.get(i)[2]+"\n";
                tabd=DataList.get(i);
                for (int j=0;j<tabd.length;j++){
                    tabline="";
                    for (int k=0;k<tabd[j].length;k++){
                        tabline+=tabd[j][k]+"\t";
                    }
                    tab+=tabline+"\n";
                }
                outstr.append(tab+"\n");
            }
            
            try {
                BufferedWriter bw=new BufferedWriter(new FileWriter("edge_table"));
                bw.write(outstr.toString());
                bw.close();
            } catch (IOException e){
                e.printStackTrace();
            }
        }
        
        
        public String[][] getEdgeData(DREM_Timeiohmm theTimeiohmm,DREM_Timeiohmm.Treenode ptr, int nchild){
                int numtf = theTimeiohmm.bindingData.regNames.length;
		int nsize = theTimeiohmm.bindingData.signedBindingValuesUnsorted.size();
		if (theTimeiohmm.bindingData.signedBindingValuesUnsorted
				.contains(new Integer(0))) {
			nsize--;
		}
		int numrows = numtf * nsize;

		boolean bsplit = ptr.numchildren >= 2;
		int nsplitoffset;
                String[] columnNames;
                String[][] tabledata;

		if (bsplit) {
			columnNames = new String[11];
			nsplitoffset = 1;
			columnNames[2] = "Num Parent";
			columnNames[7] = "Expect Split";
			columnNames[8] = "Diff. Split";
			columnNames[9] = "Score Split";
			columnNames[10] = "% Split";
		} else {
			nsplitoffset = 0;
			columnNames = new String[6];
		}

		tabledata = new String[numrows][columnNames.length];
		columnNames[0] = "TF";
		columnNames[1] = "Num Total";
		columnNames[2 + nsplitoffset] = "Num Path";
		columnNames[3 + nsplitoffset] = "Expect Overall";
		columnNames[4 + nsplitoffset] = "Diff. Overall";
		columnNames[5 + nsplitoffset] = "Score Overall";
                 
                NumberFormat nf;
                NumberFormat nf2;
		nf2 = NumberFormat.getInstance(Locale.ENGLISH);
		nf2.setMinimumFractionDigits(2);
		nf2.setMaximumFractionDigits(2);
		int nrowindex = 0;
                int ntype=1;
                
		for (int nrow = 0; nrow < numtf; nrow++) {
			for (int nel = 0; nel < theTimeiohmm.bindingData.signedBindingValuesSorted.length; nel++) {
				if (theTimeiohmm.bindingData.signedBindingValuesSorted[nel] != 0) {
					tabledata[nrowindex][0] = theTimeiohmm.bindingData.regNames[nrow]
							+ " "
							+ theTimeiohmm.bindingData.signedBindingValuesSorted[nel];
					tabledata[nrowindex][1] = ""
							+ theTimeiohmm.filteredClassifier.nBaseCount[nrow][theTimeiohmm.bindingData.signedBindingValuesSorted[nel]
									+ theTimeiohmm.filteredClassifier.noffset];

					tabledata[nrowindex][2 + nsplitoffset] = ""
							+ ptr.ncountvals[nrow][nchild][nel];
					tabledata[nrowindex][3 + nsplitoffset] = nf2
							.format(ptr.dexpectEdgeFull[nchild][nrowindex]);
					tabledata[nrowindex][4 + nsplitoffset] = nf2
							.format(ptr.ddiffEdgeFull[nchild][nrowindex]);
					tabledata[nrowindex][5 + nsplitoffset] = DREMGui_EdgeTable
							.doubleToSz(ptr.dpvalEdgeFull[nchild][nrowindex]);

					if (bsplit) {
						tabledata[nrowindex][2] = ""
								+ ptr.ncountTotals[nrow][nel];
						tabledata[nrowindex][7] = nf2
								.format(ptr.dexpectEdgeSplit[nchild][nrowindex]);
						tabledata[nrowindex][8] = nf2
								.format(ptr.ddiffEdgeSplit[nchild][nrowindex]);
						tabledata[nrowindex][9] = DREMGui_EdgeTable
								.doubleToSz(ptr.dpvalEdgeSplit[nchild][nrowindex]);
						if (ptr.ncountTotals[nrow][nel] == 0) {
							tabledata[nrowindex][10] = "0.00";
						} else {
							tabledata[nrowindex][10] = ""
									+ nf2
											.format(100
													* (double) ptr.ncountvals[nrow][nchild][nel]
													/ ptr.ncountTotals[nrow][nel]);
						}
					}
					nrowindex++;
				}
			}
		}

               
               Comparator<String[]> arrayComparator = new Comparator<String[]> (){
                    public int compare(String[] str1,String[] str2){
                        int cc;
                        if (str1.length==11){
                           cc=9;
                        }else{
                           cc=5;
                        }

                        double v1=Double.parseDouble(str1[cc]);
                        double v2=Double.parseDouble(str2[cc]);
                        if (v1-v2<0){
                            return -1;
                        }else if (v1-v2>0){
                            return 1;
                        }else {
                            return 0;
                        }
                    }
                };
                
                Arrays.sort(tabledata,arrayComparator);
                return tabledata;
        }
        
       
        
	/**
	 * Sets the colors of the individual genes
	 */
	public void setGeneColors() {
		Color currColor;

		if (ncolortime == 0) {
			for (int nrow = 0; nrow < plArray.length; nrow++) {
				plArray[nrow].setStrokePaint(lineColorsA[nrow]);
			}
		} else {
			BigInteger nval = (new BigInteger("" + theTimeiohmm.nmaxchild))
					.pow((data[0].length - ncolortime));
			// (int) Math.pow(theTimeiohmm.nmaxchild,data[0].length-ncolortime);

			for (int nrow = 0; nrow < plArray.length; nrow++) {
				// int nid = nval*(storedbestpath[nrow]/nval);
				BigInteger nid = nval.multiply(storedbestpath[nrow]
						.divide(nval));
				Integer colorIDobj = htLineIDtoColorID.get(ncolortime + ";"
						+ nid);
				if ((colorIDobj == null) || (htColorIDtoColor == null)) {
					if (DREM_Timeiohmm.BDEBUG) {
						System.out.println(ncolortime + ";" + nid + "\t"
								+ colorIDobj + "\t" + htColorIDtoColor);
					}
				}

				currColor = htColorIDtoColor.get(colorIDobj);
				plArray[nrow].setStrokePaint(currColor);
			}
		}
	}

	/**
	 * Tool tip text updater.
	 */
	private static class ToolTipTextUpdater extends PBasicInputEventHandler {
		/** Canvas. */
		private PCanvas canvas;

		/** Node. */
		private PNode node;
		String szText;

		/**
		 * Create a new tool tip text updater for the specified canvas and node.
		 * 
		 * @param canvas
		 *            canvas
		 * @param node
		 *            node
		 */
		public ToolTipTextUpdater(final PCanvas canvas, final PNode node,
				final String szText) {
			this.canvas = canvas;
			this.node = node;
			this.szText = szText;
		}

		/** @see PBasicInputEventHandler */
		public void mouseEntered(final PInputEvent e) {
			if (node.getVisible()) {
				canvas.setToolTipText(szText);
			} else {
				canvas.setToolTipText(null);
			}

		}

		/** @see PBasicInputEventHandler */
		public void mouseExited(final PInputEvent e) {
			canvas.setToolTipText(null);
		}
	}

	/**
	 * Closes the frame windows
	 */
	public void closeWindows() {
		if (filterStaticFrame != null) {
			filterStaticFrame.setVisible(false);
			filterStaticFrame.dispose();
			filterStaticFrame = null;
		}
		if (filterGOFrame != null) {
			filterGOFrame.setVisible(false);
			filterGOFrame.dispose();
			filterGOFrame = null;
		}
		if (defineFrame != null) {
			defineFrame.setVisible(false);
			defineFrame.dispose();
			defineFrame = null;
		}
		if (predictFrame != null) {
			predictFrame.setVisible(false);
			predictFrame.dispose();
			predictFrame = null;
		}
		if (keyinputFrame != null) {
			keyinputFrame.setVisible(false);
			keyinputFrame.dispose();
			keyinputFrame = null;
		}
		if (yscaleFrame != null) {
			yscaleFrame.setVisible(false);
			yscaleFrame.dispose();
			yscaleFrame = null;
		}
	}

	/**
	 * Saves the edge color selections to an ouput file
	 */
	public String saveColors() {
		StringBuffer szbuf = new StringBuffer();
		float[] f = new float[4];
		for (int nindex = 0; nindex < numcolor; nindex++) {
			Color currColor = htColorIDtoColor.get(new Integer(nindex));
			currColor.getRGBComponents(f);
			szbuf.append(f[0] + "\t" + f[1] + "\t" + f[2] + "\t" + f[3] + "\n");
		}

		return szbuf.toString();
	}

	/**
	 * Undo gene display based on the GO selection
	 */
	public void unselectGO() {
		for (int ngene = 0; ngene < bGOVisible.length; ngene++) {
			bGOVisible[ngene] = true;
			boolean bvisible = bglobalVisible && bSetVisible[ngene]
					&& bPathVisible[ngene] && bTFVisible[ngene];
			plArray[ngene].setVisible(bvisible);
			plArray[ngene].setPickable(bvisible);
		}
		bfiltergo = false;
		setFilterText();
	}

	/**
	 * Changes the gene visibility based on the selected GO category
	 * szSelectedGO
	 */
	public void selectGO(String szSelectedGO) {
		GoAnnotations tga = theTimeiohmm.theDataSet.tga;

		tga.szSelectedGO = szSelectedGO;
		String[] genenames = theTimeiohmm.theDataSet.genenames;
		for (int ngene = 0; ngene < genenames.length; ngene++) {
			HashSet<String> hsGO = tga.labelsForID(genenames[ngene]);
			if (hsGO.contains(szSelectedGO)) {
				bGOVisible[ngene] = true;
				boolean bvisible = bglobalVisible && bSetVisible[ngene]
						&& bPathVisible[ngene] && bTFVisible[ngene];
				plArray[ngene].setVisible(bvisible);
				plArray[ngene].setPickable(bvisible);
			} else {
				bGOVisible[ngene] = false;
				plArray[ngene].setVisible(false);
				plArray[ngene].setPickable(false);
			}
		}
		bfiltergo = true;
		setFilterText();
	}

	/**
	 * Generates the string in filterText which displays information as to how
	 * the genes were selected
	 */
	public void setFilterText() {
		String sz = "Genes selected based on ";

		int ncount = 0;
		if (bfiltergo)
			ncount++;
		if (bfiltergeneset)
			ncount++;
		if (bfilterinput)
			ncount++;

		boolean bpath = theSelectedRec.theCircleID != null;
		if (bpath) {
			sz += "a path constraint";

			if (ncount == 1)
				sz += " and ";
			else if (ncount >= 2)
				sz += ", ";
		}

		if (bfiltergo) {
			GoAnnotations tga = theTimeiohmm.theDataSet.tga;
			String szGO = "the GO category "
					+ ((GoAnnotations.Rec) tga.htGO.get(tga.szSelectedGO)).sztermName;
			if (bfiltergeneset) {
				if (bfilterinput) {
					sz += "a gene set, TF input, and " + szGO;
				} else {
					if (bpath) {
						sz += "a gene set, and " + szGO;
					} else {
						sz += "a gene set and " + szGO;
					}
				}
			} else {
				if (bfilterinput) {
					if (bpath) {
						sz += "TF input, and " + szGO;
					} else {
						sz += "TF input and " + szGO;
					}

				} else {
					sz += szGO;
				}
			}
		} else {
			if (bfiltergeneset) {
				if (bfilterinput) {
					if (bpath) {
						sz += "a gene set, and TF input";
					} else {
						sz += "a gene set and TF input";
					}
				} else {
					sz += "a gene set";
				}
			} else {
				if (bfilterinput) {
					sz += "TF input";
				} else {
					if (!bpath) {
						sz = "";
					}
				}
			}
		}

		filterText.setText(sz);
	}

	/**
	 * Controls the display of the tex showing the filter information
	 */
	public void renderFilterText() {
		filterText = new PText();
		setFilterText();
		filterText.translate(LEFTBUFFER + 15, 2);
		filterText.setPickable(false);
		filterText.setFont(new Font("times", Font.PLAIN, 14));
		if (!bglobalnode) {
			filterText.setVisible(false);
			filterText.setPickable(false);
		}
		canvas.getCamera().addChild(filterText);
	}

	/**
	 * Empty method
	 */
	public void componentHidden(ComponentEvent e) {
	}

	/**
	 * Empty method
	 */
	public void componentMoved(ComponentEvent e) {
	}

	/**
	 * Calls drawmain
	 */
	public void componentShown(ComponentEvent e) {
		drawmain();
	}

	/**
	 * Calls drawmain
	 */
	public void componentResized(ComponentEvent e) {
		drawmain();
	}

	/**
	 * Sets the screen size
	 */
	public void beforeInitialize() {
		if (DREM_Timeiohmm.BDEBUG) {
			System.out.println("enter beforeinitialize");
		}
		setSize(SCREENWIDTH, SCREENHEIGHT);
		if (DREM_Timeiohmm.BDEBUG) {
			System.out.println("leave beforeinitialize");
		}
	}

	/**
	 * calls drawmain
	 */
	public void initialize() {
		drawmain();
	}

	/**
	 * Responsible for laying out the main interface window
	 */
	public void drawmain() {
		synchronized (datasetlock) {
			while (!binit) {
				try {
					datasetlock.wait();
				} catch (InterruptedException e) {
				}
			}

			hideList = new ArrayList<PNode>();
			ncircleID = 0;
			numcolor = 0;
			nparentcolorindex = 0;
			htColorIDtoLinesList = new Hashtable<Integer, ArrayList<PPath>>();
			circleSet = new TreeSet<CircleRec>(new CircleRecCompare());

			htColors = new Hashtable<PNode, Color>();
			htNodes = new Hashtable<PNode, CircleID>();
			htPathToLineSet = new Hashtable<BigInteger, ArrayList<Integer>>();
			if (hidesigList != null) {
				htTextVisible = new Hashtable<String, Boolean>();
				int nsize = hidesigList.size();
				for (int nindex = 0; nindex < nsize; nindex++) {
					SigInfoRec theSigInfoRec = hidesigList.get(nindex);
					// htTextVisible.put(theSigInfoRec.ndepth + ";"
					// + theSigInfoRec.npathscore + ";"
					// + theSigInfoRec.ntype, Boolean
					// .valueOf(theSigInfoRec.theSigText.getVisible()));
					htTextVisible.put(theSigInfoRec.ndepth + ";"
							+ theSigInfoRec.npathscore + ";"
							+ theSigInfoRec.ntype, Boolean
							.valueOf(theSigInfoRec.border.getChild(0)
									.getVisible()));
				}
			} else {
				htTextVisible = null;
			}
			hidesigList = new ArrayList<SigInfoRec>();
			hidegolabelsList = new ArrayList<PText>();
			hidetfsetlabelsList = new ArrayList<PText>();
			hidegenesetlabelsList = new ArrayList<PText>();
			hidepredictlabelsList = new ArrayList<PText>();

			canvas = getCanvas();
			ToolTipManager.sharedInstance().registerComponent(canvas);
			theLayer = canvas.getLayer();
			PCamera theCamera = canvas.getCamera();
			theLayer.removeAllChildren();
			theCamera.removeAllChildren();

			dmin = dinitmin / dscaley;
			dmax = dinitmax / dscaley;
			double dmaxmindiff = (dmax - dmin);

			nheight = getHeight();
			REALHEIGHT = (nheight - TOPBUFFER - BOTTOMBUFFER);
			dheightunits = REALHEIGHT / dmaxmindiff;
			nwidth = getWidth();

			int numintervals = data[0].length - 1;
			dwidthunits = (nwidth - LEFTBUFFER - RIGHTBUFFER) * dscalex;

			dwidthunitsInterval = new double[numintervals];
			dwidthunitsCum = new double[numintervals + 1];
			dwidthunitsCum[0] = 0;

			boolean bincreasing = true;

			try {
				double dprev = Util
						.removeUnits(theTimeiohmm.theDataSet.dsamplemins[0]);
				for (int ni = 0; ni < dwidthunitsInterval.length; ni++) {
					double dnext = Util
							.removeUnits(theTimeiohmm.theDataSet.dsamplemins[ni + 1]);
					dwidthunitsInterval[ni] = dnext - dprev;
					if (dwidthunitsInterval[ni] < 0) {
						bincreasing = false;
						brealXaxis = false;
						binvalidreal = true;
					}
					dwidthunitsCum[ni + 1] = dwidthunitsCum[ni]
							+ dwidthunitsInterval[ni];
					dprev = dnext;
				}
				double dtotal = dwidthunitsCum[dwidthunitsCum.length - 1];
				for (int ni = 1; ni < dwidthunitsCum.length; ni++) {
					dwidthunitsInterval[ni - 1] = dwidthunits
							* dwidthunitsInterval[ni - 1] / dtotal;
					dwidthunitsCum[ni] = dwidthunits * dwidthunitsCum[ni]
							/ dtotal;
				}
			} catch (IllegalArgumentException iae) {
				binvalidreal = true;
				brealXaxis = false;
				for (int ni = 1; ni < dwidthunitsCum.length; ni++) {
					dwidthunitsInterval[ni - 1] = dwidthunits / numintervals;
					dwidthunitsCum[ni] = ni * dwidthunits / numintervals;
				}
			}

			if ((!brealXaxis) || (!bincreasing)) {
				for (int ni = 1; ni < dwidthunitsCum.length; ni++) {
					dwidthunitsInterval[ni - 1] = dwidthunits / numintervals;
					dwidthunitsCum[ni] = ni * dwidthunits / numintervals;
				}
			}

			PLine plxaxis = new PLine();
			double doriginy = REALHEIGHT + dheightunits * dmin;
			plxaxis.addPoint(0, LEFTBUFFER, doriginy);
			plxaxis.addPoint(1, dscalex * nwidth - RIGHTBUFFER / 2, doriginy);
			plxaxis.setStroke(new BasicStroke(4));
			plxaxis.setStrokePaint(Color.gray);
			theLayer.addChild(plxaxis);

			for (int ncol = 0; ncol < theTimeiohmm.theDataSet.dsamplemins.length; ncol++) {
				PLine pltick = new PLine();
				double dxp = dwidthunitsCum[ncol] + LEFTBUFFER;
				pltick.addPoint(0, dxp, doriginy);
				pltick.addPoint(1, dxp, doriginy + TICKLENGTH);
				pltick.setStroke(new BasicStroke(4));
				pltick.setStrokePaint(Color.gray);
				theLayer.addChild(pltick);

				PText tickLabel = new PText(""
						+ theTimeiohmm.theDataSet.dsamplemins[ncol]);
				tickLabel.translate(dxp + 4, doriginy + 4);
				theLayer.addChild(tickLabel);
			}

			int npower = (int) Math.floor(Math.log(Math.max(Math.abs(dmin),
					dmax))
					/ Math.log(10));
			NumberFormat nftick = NumberFormat.getInstance(Locale.ENGLISH);
			if (npower < 0) {
				nftick.setMaximumFractionDigits(-npower);
				nftick.setMinimumFractionDigits(-npower);
			}

			double drealincrementy = Math.pow(10, npower);
			double dincrementy = drealincrementy * dheightunits;
			double drealy = -drealincrementy;
			double dticky = doriginy + dincrementy;

			while (drealy < dinitmax) {
				dticky -= dincrementy;
				drealy += drealincrementy;
				PLine pltick = new PLine();
				pltick.addPoint(0, LEFTBUFFER - TICKLENGTH, dticky);
				pltick.addPoint(1, LEFTBUFFER, dticky);
				pltick.setStroke(new BasicStroke(4));
				pltick.setStrokePaint(Color.gray);
				theLayer.addChild(pltick);
				String sztickval = nftick.format(drealy);
				PText tickLabel = new PText(sztickval);

				tickLabel.translate(LEFTBUFFER - TICKLENGTH - 4
						* sztickval.length() + 6, dticky - 17);
				theLayer.addChild(tickLabel);
			}

			PLine plyaxis = new PLine();
			plyaxis.addPoint(0, LEFTBUFFER, dticky);

			dticky = doriginy;
			drealy = 0;

			while (drealy > dinitmin) {
				dticky += dincrementy;
				drealy -= drealincrementy;
				PLine pltick = new PLine();
				pltick.addPoint(0, LEFTBUFFER - TICKLENGTH, dticky);
				pltick.addPoint(1, LEFTBUFFER, dticky);
				pltick.setStroke(new BasicStroke(4));
				pltick.setStrokePaint(Color.gray);
				theLayer.addChild(pltick);
				String sztickval = nftick.format(drealy);
				PText tickLabel = new PText(sztickval);
				tickLabel.translate(LEFTBUFFER - TICKLENGTH - 3
						* sztickval.length() + 3, dticky - 17);
				theLayer.addChild(tickLabel);
			}

			plyaxis.addPoint(1, LEFTBUFFER, dticky);
			plyaxis.setStroke(new BasicStroke(4));
			plyaxis.setStrokePaint(Color.gray);
			theLayer.addChild(plyaxis);

			Random lineColors = new Random(74312);
			if (lineColorsA == null) {
				lineColorsA = new Color[data.length];
			}

			for (int nrow = 0; nrow < data.length; nrow++) {
				Color lineColor;

				if (lineColorsA[nrow] == null) {

					lineColor = new Color(lineColors.nextInt(226), lineColors
							.nextInt(226), lineColors.nextInt(226));
					lineColorsA[nrow] = lineColor;
				}

				PLine pl = new PLine();

				pl.addInputEventListener(new ToolTipTextUpdater(canvas, pl,
						genenames[nrow]));

				int nadded = 0;
				for (int ncol = 0; ncol < data[nrow].length; ncol++) {
					if (pmavalues[nrow][ncol] != 0) {
						double dxp = dwidthunitsCum[ncol] + LEFTBUFFER;
						double dyp = REALHEIGHT - dheightunits
								* (data[nrow][ncol] - dmin);
						pl.addPoint(nadded, dxp, dyp);
						nadded++;
					}
				}

				ArrayList<Integer> geneList = htPathToLineSet
						.get(storedbestpath[nrow]);

				if (geneList == null) {
					geneList = new ArrayList<Integer>();
				}

				geneList.add(new Integer(nrow));
				htPathToLineSet.put(storedbestpath[nrow], geneList);
				plArray[nrow] = pl;
				boolean bvisible = bglobalVisible && bPathVisible[nrow]
						&& bTFVisible[nrow] && bGOVisible[nrow]
						&& bSetVisible[nrow];

				pl.setVisible(bvisible);
				pl.setPickable(bvisible);
				theLayer.addChild(pl);
			}

			drawNodes(treecopy, 0, 0, 0, new BigInteger("" + 0), 0, null, null,
					-1, nparentcolorindex, treecopy.nminparentlevel);
			setGeneColors();
			Iterator<CircleRec> circleItr = circleSet.iterator();
			while (circleItr.hasNext()) {
				CircleRec theCircleRec = circleItr.next();
				theLayer.addChild(theCircleRec.circle);

				if (!bglobalnode) {
					theCircleRec.circle.setVisible(false);
					theCircleRec.circle.setPickable(false);
				}
			}

			int npredictsize = hidepredictlabelsList.size();
			for (int nindex = 0; nindex < npredictsize; nindex++) {
				theLayer.addChild(hidepredictlabelsList.get(nindex));
			}

			int ntfsetsize = hidetfsetlabelsList.size();
			for (int nindex = 0; nindex < ntfsetsize; nindex++) {
				theLayer.addChild(hidetfsetlabelsList.get(nindex));
			}

			int ngenesetsize = hidegenesetlabelsList.size();
			for (int nindex = 0; nindex < ngenesetsize; nindex++) {
				theLayer.addChild(hidegenesetlabelsList.get(nindex));
			}

			int ngosize = hidegolabelsList.size();
			for (int nindex = 0; nindex < ngosize; nindex++) {
				theLayer.addChild(hidegolabelsList.get(nindex));
			}

			genetableButton = PPath.createRectangle((float) 0.0, (float) 0.0,
					(float) BUTTONRECWIDTH, (float) 18.0);
			PText genetableText = new PText("Gene Table");
			genetableText.setFont(new Font("times", Font.PLAIN, 12));
			genetableText.translate(26, 2);
			genetableButton.setPaint(buttonColor);
			genetableButton.addChild(genetableText);

			genetableButton
					.addInputEventListener(new PBasicInputEventHandler() {
						public void mousePressed(PInputEvent event) {
							javax.swing.SwingUtilities
									.invokeLater(new Runnable() {
										public void run() {
											JFrame frame = new JFrame(
													"Table of Selected Genes");
											frame
													.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
											frame.setLocation(20, 50);
											boolean[] bdisplay = new boolean[plArray.length];
											for (int ngene = 0; ngene < plArray.length; ngene++) {
												bdisplay[ngene] = bPathVisible[ngene]
														&& bTFVisible[ngene]
														&& bGOVisible[ngene]
														&& bSetVisible[ngene];
											}
											DREMGui_GeneTable newContentPane = new DREMGui_GeneTable(
													frame,
													theTimeiohmm.theDataSet,
													theTimeiohmm,
													theTimeiohmm.bindingData.gene2RegBinding,
													theTimeiohmm.bindingData.gene2RegBindingIndex,
													theTimeiohmm.bindingData.reg2GeneBinding,
													theTimeiohmm.bindingData.reg2GeneBindingIndex,
													theTimeiohmm.bindingData.gene2RegMaxBinding,
													theTimeiohmm.bindingData.gene2RegMaxBindingIndex,
													theTimeiohmm.bindingData.regNames,
													bdisplay);
											// need to bindingpvalGeneIndex
											// reference
											newContentPane.setOpaque(true); // content
											// panes
											// must
											// be
											// opaque
											frame
													.setContentPane(newContentPane);
											// Display the window.
											frame.pack();
											frame.setVisible(true);
										}
									});
						}
					});

			final DREMGui ftheDREMGui = this;

			setButton = PPath.createRectangle((float) 0.0, (float) 0.0,
					(float) BUTTONRECWIDTH, (float) 18.0);

			PText setText = new PText("Select by Gene Set");
			setText.setFont(new Font("times", Font.PLAIN, 12));
			setText.translate(4, 2);
			setButton.setPaint(buttonColor);
			setButton.addChild(setText);

			setButton.addInputEventListener(new PBasicInputEventHandler() {
				public void mousePressed(PInputEvent event) {
					javax.swing.SwingUtilities.invokeLater(new Runnable() {
						public void run() {
							if (defineFrame == null) {
								defineFrame = new JFrame(
										"Select Genes Based on Defined Gene Set");
								defineFrame
										.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
								defineFrame.setLocation(400, 200);
								DREMGui_DefineGeneSet newContentPane = new DREMGui_DefineGeneSet(
										defineFrame,
										theTimeiohmm.theDataSet.tga,
										ftheDREMGui, ftheDREMGui.treecopy);
								newContentPane.setOpaque(true);
								// content panes must be opaque
								defineFrame.setContentPane(newContentPane);

								// Display the window.
								defineFrame.pack();
							} else {
								defineFrame.setExtendedState(Frame.NORMAL);
							}
							defineFrame.setVisible(true);
						}
					});
				}
			});

			gotableButton = PPath.createRectangle((float) 0.0, (float) 0.0,
					(float) BUTTONRECWIDTH, (float) 18.0);
			PText gotableText = new PText("GO Table");
			gotableText.setFont(new Font("times", Font.PLAIN, 12));
			gotableText.translate(30, 2);
			gotableButton.setPaint(buttonColor);
			gotableButton.addChild(gotableText);

			gotableButton.addInputEventListener(new PBasicInputEventHandler() {
				public void mousePressed(PInputEvent event) {
					javax.swing.SwingUtilities.invokeLater(new Runnable() {
						public void run() {
							JFrame frame = new JFrame(
									"GO Enrichment for Selected Genes");
							frame
									.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
							frame.setLocation(20, 50);
							frame.setSize(826, 509);
							frame.setVisible(true);

							boolean[] bdisplay = new boolean[plArray.length];
							for (int ngene = 0; ngene < plArray.length; ngene++) {
								bdisplay[ngene] = bPathVisible[ngene]
										&& bTFVisible[ngene]
										&& bGOVisible[ngene]
										&& bSetVisible[ngene];
							}
							DREMGui_GOTable newContentPane = new DREMGui_GOTable(
									ftheDREMGui,
									frame,
									theTimeiohmm.theDataSet,
									theTimeiohmm.bindingData.gene2RegMaxBinding,
									theTimeiohmm.bindingData.gene2RegMaxBindingIndex,
									theTimeiohmm.bindingData.reg2GeneMaxBinding,
									theTimeiohmm.bindingData.reg2GeneMaxBindingIndex,
									theTimeiohmm.bindingData.regNames, bdisplay);

							// need to add geneIndex here

							newContentPane.setOpaque(true); // content panes
							// must be opaque
							frame.setContentPane(newContentPane);
							// Display the window.
							frame.pack();
						}
					});
				}
			});

			staticButton = PPath.createRectangle((float) 0.0, (float) 0.0,
					(float) BUTTONRECWIDTH, (float) 18.0);
			PText staticText = new PText("Select by TFs");
			staticText.setFont(new Font("times", Font.PLAIN, 12));
			staticText.translate(14, 2);
			staticButton.setPaint(buttonColor);
			staticButton.addChild(staticText);

			staticButton.addInputEventListener(new PBasicInputEventHandler() {
				public void mousePressed(PInputEvent event) {
					javax.swing.SwingUtilities.invokeLater(new Runnable() {
						public void run() {
							if (filterStaticFrame == null) {
								filterStaticFrame = new JFrame(
										"Select Genes based on TF Input Constraints");
								filterStaticFrame
										.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
								filterStaticFrame.setLocation(500, 200);
								DREMGui_FilterStatic newContentPane = new DREMGui_FilterStatic(
										filterStaticFrame, ftheDREMGui,
										theTimeiohmm.theDataSet.tga,
										ftheDREMGui.treecopy);
								newContentPane.setOpaque(true);
								// content panes must be opaque
								filterStaticFrame
										.setContentPane(newContentPane);

								// Display the window.
								filterStaticFrame.pack();
							} else {
								filterStaticFrame
										.setExtendedState(Frame.NORMAL);
							}
							filterStaticFrame.setVisible(true);
						}
					});
				}
			});

			gofilterButton = PPath.createRectangle((float) 0.0, (float) 0.0,
					(float) BUTTONRECWIDTH, (float) 18.0);

			gofilterButton.addInputEventListener(new PBasicInputEventHandler() {
				public void mousePressed(PInputEvent event) {
					javax.swing.SwingUtilities.invokeLater(new Runnable() {
						public void run() {
							if (filterGOFrame == null) {
								filterGOFrame = new JFrame(
										"Select Genes based on GO Category");
								filterGOFrame
										.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
								filterGOFrame.setLocation(500, 250);

								theGOFilter = new DREMGui_GOFilter(
										filterGOFrame,
										ftheDREMGui,
										ftheDREMGui.theTimeiohmm.theDataSet.tga.theRecIDdrem,
										ftheDREMGui.treecopy);
								theGOFilter.setOpaque(true);
								filterGOFrame.setContentPane(theGOFilter);

								// Display the window.
								filterGOFrame.pack();
							} else {
								filterGOFrame.setExtendedState(Frame.NORMAL);
							}
							filterGOFrame.setVisible(true);
						}

					});
				}
			});

			PText gofilterText = new PText("Select by GO");
			gofilterText.setFont(new Font("times", Font.PLAIN, 12));
			gofilterText.translate(22, 2);
			gofilterButton.setPaint(buttonColor);
			gofilterButton.addChild(gofilterText);

			scaleButton = PPath.createRectangle((float) 0.0, (float) 0.0,
					(float) BUTTONRECWIDTH, (float) 18.0);

			PText scaleText = new PText("Interface Options");
			scaleText.setFont(new Font("times", Font.PLAIN, 12));
			scaleText.translate(10, 2);
			scaleButton.setPaint(buttonColor);
			scaleButton.addChild(scaleText);

			scaleButton.addInputEventListener(new PBasicInputEventHandler() {
				public void mousePressed(PInputEvent event) {

					javax.swing.SwingUtilities.invokeLater(new Runnable() {
						public void run() {
							if (yscaleFrame == null) {
								yscaleFrame = new JFrame("Interface Options");
								yscaleFrame
										.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
								yscaleFrame.setLocation(650, 150);
								theYScalegui = new DREMGui_InterfaceOptions(
										yscaleFrame, ftheDREMGui);
								theYScalegui.setOpaque(true);
								// content panes must be opaque
								yscaleFrame.setContentPane(theYScalegui);
								// Display the window.
								yscaleFrame.pack();
							} else {
								yscaleFrame.setExtendedState(Frame.NORMAL);
							}
							yscaleFrame.setVisible(true);
						}
					});
				}
			});

			predictButton = PPath.createRectangle((float) 0.0, (float) 0.0,
					(float) BUTTONRECWIDTH, (float) 18.0);

			PText predictText = new PText("Predict");
			predictText.setFont(new Font("times", Font.PLAIN, 12));
			predictText.translate(35, 2);
			predictButton.setPaint(buttonColor);
			predictButton.addChild(predictText);

			predictButton.addInputEventListener(new PBasicInputEventHandler() {
				public void mousePressed(PInputEvent event) {
					javax.swing.SwingUtilities.invokeLater(new Runnable() {
						public void run() {
							if (predictFrame == null) {
								predictFrame = new JFrame(
										"Predict Time Series based on TF Input");
								predictFrame
										.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
								predictFrame.setLocation(400, 300);
								DREMGui_Predict newContentPane = new DREMGui_Predict(
										predictFrame, ftheDREMGui,
										ftheDREMGui.treecopy);
								newContentPane.setOpaque(true);
								// content panes must be opaque
								predictFrame.setContentPane(newContentPane);

								// Display the window.
								predictFrame.pack();
							} else {
								predictFrame.setExtendedState(Frame.NORMAL);
							}
							predictFrame.setVisible(true);
						}
					});
				}
			});

			hideButton = PPath.createRectangle((float) 0.0, (float) 0.0,
					(float) BUTTONRECWIDTH, (float) 18.0);

			if (bglobalnode) {
				hideText = new PText("Hide Nodes");
			} else {
				hideText = new PText("Show Nodes");
			}
			hideText.setFont(new Font("times", Font.PLAIN, 12));
			hideText.translate(23, 2);
			hideButton.setPaint(buttonColor);
			hideButton.addChild(hideText);

			hideButton.addInputEventListener(new PBasicInputEventHandler() {
				public void mousePressed(PInputEvent event) {

					int nsize = hideList.size();

					if (bglobalnode) {
						hideText.setText("Show Nodes");
						for (int nindex = 0; nindex < nsize; nindex++) {
							PNode theNode = hideList.get(nindex);
							theNode.setVisible(false);
							theNode.setPickable(false);
						}

						if (battachlabels) {
							hidelabels();
						}

						bglobalnode = false;
					} else {
						hideText.setText("Hide Nodes");
						for (int nindex = 0; nindex < nsize; nindex++) {
							PNode theNode = hideList.get(nindex);
							theNode.setVisible(true);
							theNode.setPickable(true);
						}

						if (battachlabels) {
							showlabels();
						}
						bglobalnode = true;
					}

				}
			});

			saveDREMButton = PPath.createRectangle((float) 0.0, (float) 0.0,
					(float) BUTTONRECWIDTH, (float) 18.0);

			PText saveDREMText;
			saveDREMText = new PText(" Save Image ");
			saveDREMText.setFont(new Font("times", Font.PLAIN, 12));
			saveDREMText.translate(18, 2);
			saveDREMButton.setPaint(buttonColor);
			saveDREMButton.addChild(saveDREMText);

			saveDREMButton.addInputEventListener(new PBasicInputEventHandler() {
				public void mousePressed(PInputEvent event) {
					javax.swing.SwingUtilities.invokeLater(new Runnable() {
						public void run() {
							if (saveDREMFrame == null) {
								saveDREMFrame = new JFrame("Save as Image");
								saveDREMFrame
										.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
								saveDREMFrame.setLocation(400, 300);
								DREMGui_SaveDREM newContentPane = new DREMGui_SaveDREM(
										ftheDREMGui, saveDREMFrame);
								newContentPane.setOpaque(true);
								// content panes must be opaque
								saveDREMFrame.setContentPane(newContentPane);
								// Display the window.
								saveDREMFrame.pack();
							} else {
								saveDREMFrame.setExtendedState(Frame.NORMAL);
							}
							saveDREMFrame.setVisible(true);
						}
					});
				}
			});

			saveModelButton = PPath.createRectangle((float) 0.0, (float) 0.0,
					(float) BUTTONRECWIDTH, (float) 18.0);

			PText saveModelText;
			saveModelText = new PText(" Save Model ");
			saveModelText.setFont(new Font("times", Font.PLAIN, 12));
			saveModelText.translate(18, 2);
			saveModelButton.setPaint(buttonColor);
			saveModelButton.addChild(saveModelText);

			saveModelButton
					.addInputEventListener(new PBasicInputEventHandler() {
						public void mousePressed(PInputEvent event) {
							javax.swing.SwingUtilities
									.invokeLater(new Runnable() {
										public void run() {
											if (saveModelFrame == null) {
												saveModelFrame = new JFrame(
														"Save Model to File");
												saveModelFrame
														.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
												saveModelFrame.setLocation(400,
														300);
												DREMGui_SaveModel newContentPane = new DREMGui_SaveModel(
														ftheDREMGui.theTimeiohmm,
														treecopy,
														saveModelFrame,
														ftheDREMGui);
												newContentPane.setOpaque(true);
												// content panes must be opaque
												saveModelFrame
														.setContentPane(newContentPane);
												// Display the window.
												saveModelFrame.pack();
											} else {
												saveModelFrame
														.setExtendedState(Frame.NORMAL);
											}
											saveModelFrame.setVisible(true);
										}
									});
						}
					});

			timeButton = PPath.createRectangle((float) 0.0, (float) 0.0,
					(float) BUTTONRECWIDTH, (float) 18.0);

			PText timeText;
			if (bglobalVisible) {
				timeText = new PText("Hide Time Series");
			} else {
				timeText = new PText("Show Time Series");
			}
			timeText.setFont(new Font("times", Font.PLAIN, 12));
			timeText.translate(7, 2);
			timeButton.setPaint(buttonColor);
			timeButton.addChild(timeText);

			final PText ftimeText = timeText;
			final PNode[] fplArray = plArray;
			timeButton.addInputEventListener(new PBasicInputEventHandler() {
				public void mousePressed(PInputEvent event) {
					if (bglobalVisible) {
						ftimeText.setText("Show Time Series");
						for (int nindex = 0; nindex < fplArray.length; nindex++) {
							fplArray[nindex].setVisible(false);
							fplArray[nindex].setPickable(false);
						}
						bglobalVisible = false;
					} else {
						ftimeText.setText("Hide Time Series");
						for (int nindex = 0; nindex < fplArray.length; nindex++) {
							boolean bvisible = bPathVisible[nindex]
									&& bTFVisible[nindex] && bGOVisible[nindex]
									&& bSetVisible[nindex];
							fplArray[nindex].setVisible(bvisible);
							fplArray[nindex].setPickable(bvisible);
						}
						bglobalVisible = true;
					}
				}
			});

			siginButton = PPath.createRectangle((float) 0.0, (float) 0.0,
					(float) BUTTONRECWIDTH, (float) 18.0);

			PText siginText = new PText("Key TFs Labels");
			siginText.setFont(new Font("times", Font.PLAIN, 12));
			siginText.translate(10, 2);
			siginButton.setPaint(buttonColor);
			siginButton.addChild(siginText);

			nheight = getHeight();
			int ninset = 10;
			int nspacing = ninset + BUTTONRECWIDTH;

			nwidth = getWidth();
			INITLEFT = nwidth / 2.0 - 3 * nspacing;
			helpButton = new PImage(Util.getImageURL("Help24.gif"));
			PImage saveButton = new PImage(Util.getImageURL("Save24.gif"));

			final DREMGui thisFrame = this;

			saveButton.addInputEventListener(new PBasicInputEventHandler() {
				public void mousePressed(PInputEvent event) {
					try {
						int nreturnVal = Util.theChooser
								.showSaveDialog(thisFrame);
						if (nreturnVal == JFileChooser.APPROVE_OPTION) {
							File f = Util.theChooser.getSelectedFile();
							PrintWriter pw = new PrintWriter(
									new FileOutputStream(f));
							printDefaults(pw);
							pw.close();
						}
					} catch (FileNotFoundException ex) {
						final FileNotFoundException fex = ex;
						javax.swing.SwingUtilities.invokeLater(new Runnable() {
							public void run() {
								JOptionPane.showMessageDialog(null, fex
										.getMessage(), "Exception thrown",
										JOptionPane.ERROR_MESSAGE);
							}
						});
						ex.printStackTrace(System.out);
					} finally {
						// Fixes a bug in which the DREM model is dragged automatically
						// after closing the save dialog
						event.setHandled(true);
					}
				}
			});

			helpButton.addInputEventListener(new PBasicInputEventHandler() {
				public void mousePressed(PInputEvent event) {
					javax.swing.SwingUtilities.invokeLater(new Runnable() {
						public void run() {
							JDialog helpDialog = new JDialog(thisFrame, "Help",
									false);
							Container theHelpDialogPane = helpDialog
									.getContentPane();

							helpDialog.setBackground(Color.white);
							theHelpDialogPane.setBackground(Color.white);
							String szMessage = "This is the main interface window of DREM. "
									+ "The window shows a plot of the expression profiles of genes that were "
									+ "not filtered along with the map learned by DREM, and associated TF labels.  "
									+ "Green nodes are split nodes, and nodes size is proportional to the standard deviation of their assocoated "
									+ "gaussian emission distribution.  "
									+ "Left clicking on an edge of the map shows only genes assigned to that path through the model.  Right clicking "
									+ "on an edge on TF label box brings up info about the TFs regulating genes assigned to the path.\n\n"
									+ "The buttons along the bottom function as follows: \n"
									+ "Predict - displays for a given TF input vector the probability of transitioning to each state \n"
									+ "Interface Options - displays menu to adjust interface options \n"
									+ "GO Table - displays a GO enrichment analysis for the currently selected genes\n"
									+ "Gene Table - displays a gene table for the currently selected genes\n"
									+ "Key TFs Labels - adjust the criteria and threshold for display TF labels on the map\n"
									+ "Select by TFs - select genes by which TFs they regulated by \n"
									+ "Select by GO - selects genes based on the GO category to which they are annotated\n"
									+ "Select by Gene Set - selects genes based on a defined set\n"
									+ "Hide Nodes/Show Nodes - hide/shows the nodes \n"
									+ "Hide Time Series/Show Time Series - hide/shows the time series expression patterns on the interface\n"
									+ "Save Model - exports the model to a file can then be loaded later under the 'Saved Model File' option\n"
									+ "Save Image - saves the interface window to a graphics file\n"
									+ "Save Disk - saves model parameters used to generate the current results out to a default text file.";

							JTextArea textArea = new JTextArea(szMessage, 10,
									60);
							textArea.setLineWrap(true);
							textArea.setWrapStyleWord(true);
							textArea.setBackground(Color.white);
							textArea.setEditable(false);

							JScrollPane jsp2 = new JScrollPane(textArea);

							theHelpDialogPane.add(jsp2);
							theHelpDialogPane.setSize(820, 600);
							theHelpDialogPane.validate();

							helpDialog.setLocation(thisFrame.getX() + 50,
									thisFrame.getY() + 25);

							helpDialog.setSize(820, 600);
							helpDialog
									.setDefaultCloseOperation(DISPOSE_ON_CLOSE);
							helpDialog.setVisible(true);
						}
					});
				}
			});

			theCamera.addChild(saveDREMButton);
			theCamera.addChild(saveModelButton);
			theCamera.addChild(helpButton);
			theCamera.addChild(saveButton);
			theCamera.addChild(staticButton);
			theCamera.addChild(setButton);
			theCamera.addChild(gofilterButton);
			theCamera.addChild(scaleButton);
			theCamera.addChild(predictButton);
			theCamera.addChild(hideButton);
			theCamera.addChild(timeButton);
			theCamera.addChild(siginButton);
			theCamera.addChild(genetableButton);
			theCamera.addChild(gotableButton);

			scaleButton.translate(INITLEFT, nheight - 75);
			predictButton.translate(INITLEFT, nheight - 97);

			genetableButton.translate(INITLEFT + nspacing, nheight - 75);
			gotableButton.translate(INITLEFT + nspacing, nheight - 97);

			siginButton.translate(INITLEFT + 2 * nspacing, nheight - 97);
			staticButton.translate(INITLEFT + 2 * nspacing, nheight - 75);

			setButton.translate(INITLEFT + 3 * nspacing, nheight - 75);
			gofilterButton.translate(INITLEFT + 3 * nspacing, nheight - 97);

			helpButton.translate(INITLEFT + 6 * nspacing, nheight - 79);
			saveButton.translate(INITLEFT + 6 * nspacing, nheight - 101);

			saveDREMButton.translate(INITLEFT + 5 * nspacing, nheight - 75);
			saveModelButton.translate(INITLEFT + 5 * nspacing, nheight - 97);

			hideButton.translate(INITLEFT + 4 * nspacing, nheight - 97);
			timeButton.translate(INITLEFT + 4.0 * nspacing, nheight - 75);

			siginButton.addInputEventListener(new PBasicInputEventHandler() {
				public void mousePressed(PInputEvent event) {
					javax.swing.SwingUtilities.invokeLater(new Runnable() {
						public void run() {
							if (keyinputFrame == null) {
								keyinputFrame = new JFrame(
										"Key Transcription Factors (TFs) Labels");
								keyinputFrame
										.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
								keyinputFrame.setLocation(600, 350);
								DREMGui_KeyInputs newContentPane = new DREMGui_KeyInputs(
										keyinputFrame, ftheDREMGui);
								newContentPane.setOpaque(true);
								// content panes must be opaque
								keyinputFrame.setContentPane(newContentPane);

								// Display the window.
								keyinputFrame.pack();
							} else {
								keyinputFrame.setExtendedState(Frame.NORMAL);
							}
							keyinputFrame.setVisible(true);
						}
					});
				}
			});

			int nsize = hidesigList.size();
			for (int nindex = 0; nindex < nsize; nindex++) {
				SigInfoRec theSigInfoRec = hidesigList.get(nindex);

				theLayer.addChild(theSigInfoRec.border);

				double dprewidth = theSigInfoRec.border.getWidth();
                                System.out.println(nindex);
				setSigText(theSigInfoRec.theSigTF,
						(PPath) theSigInfoRec.border, theSigInfoRec.ntype,
						theSigInfoRec.ndepth, displayTF, displayMIRNA,
						theSigInfoRec.node, colorSigTFs);
				theSigInfoRec.border.translate(dprewidth
						- theSigInfoRec.border.getWidth(), 0);

			}
			renderFilterText();

		} // synchronized
	}

	/**
	 * Hides all annotation labels
	 */
	public void hidelabels() {
		int ngolabelsize = hidegolabelsList.size();
		for (int nindex = 0; nindex < ngolabelsize; nindex++) {
			PText theText = hidegolabelsList.get(nindex);
			theText.setVisible(false);
			theText.setPickable(false);
		}

		int ngenesetlabelsize = hidegenesetlabelsList.size();
		for (int nindex = 0; nindex < ngenesetlabelsize; nindex++) {
			PText theText = hidegenesetlabelsList.get(nindex);
			theText.setVisible(false);
			theText.setPickable(false);
		}

		int ntflabelsize = hidetfsetlabelsList.size();
		for (int nindex = 0; nindex < ntflabelsize; nindex++) {
			PText theText = hidetfsetlabelsList.get(nindex);
			theText.setVisible(false);
			theText.setPickable(false);
		}

		int npredictsize = hidepredictlabelsList.size();
		for (int nindex = 0; nindex < npredictsize; nindex++) {
			PText theText = hidepredictlabelsList.get(nindex);
			theText.setVisible(false);
			theText.setPickable(false);
		}

		int nsizesig = hidesigList.size();
		for (int nindex = 0; nindex < nsizesig; nindex++) {
			DREMGui.SigInfoRec theSigInfoRec = hidesigList.get(nindex);
			PPath rect = (PPath) theSigInfoRec.border;
			rect.setVisible(false);
			rect.setPickable(false);

			for (Object osmallrect : rect.getChildrenReference()) {
				PNode smallrect = ((PNode) osmallrect);
				smallrect.setVisible(false);
				smallrect.setPickable(false);
				for (Object otext : smallrect.getChildrenReference()) {
					PNode text = (PNode) otext;
					text.setVisible(false);
					text.setPickable(false);
				}
			}
		}

		filterText.setVisible(false);
		filterText.setPickable(false);
	}

	/**
	 * Shows all annotation labels
	 */
	public void showlabels() {
		if ((bshowgolabels) && (bapplygolabels)) {
			int ngolabelsize = hidegolabelsList.size();
			for (int nindex = 0; nindex < ngolabelsize; nindex++) {
				PText theText = hidegolabelsList.get(nindex);
				theText.setVisible(true);
			}
		}

		if ((bshowgenesetlabels) && (bapplygenesetlabels)) {
			int ngenesetlabelsize = hidegenesetlabelsList.size();
			for (int nindex = 0; nindex < ngenesetlabelsize; nindex++) {
				PText theText = hidegenesetlabelsList.get(nindex);
				theText.setVisible(true);
			}
		}

		if ((bshowtfsetlabels) && (bapplytfsetlabels)) {
			int ntflabelsize = hidetfsetlabelsList.size();
			for (int nindex = 0; nindex < ntflabelsize; nindex++) {
				PText theText = hidetfsetlabelsList.get(nindex);
				theText.setVisible(true);
			}
		}

		if (bshowpredict) {
			int npredictsize = hidepredictlabelsList.size();
			for (int nindex = 0; nindex < npredictsize; nindex++) {
				PText theText = hidepredictlabelsList.get(nindex);
				theText.setVisible(true);
			}
		}

		if (bshowkeyinputs) {
			int nsizesig = hidesigList.size();
			for (int nindex = 0; nindex < nsizesig; nindex++) {
				DREMGui.SigInfoRec theSigInfoRec = hidesigList.get(nindex);
				PPath rect = (PPath) theSigInfoRec.border;
				rect.setVisible(true);
				rect.setPickable(true);
				for (Object osmallrect : rect.getChildrenReference()) {
					PNode smallrect = ((PNode) osmallrect);
					smallrect.setVisible(true);
					smallrect.setPickable(true);
					for (Object otext : smallrect.getChildrenReference()) {
						PNode text = (PNode) otext;
						text.setVisible(true);
						text.setPickable(true);
					}
				}
			}
		}

		filterText.setVisible(true);
		filterText.setPickable(false);
	}

	/**
	 * Information about circle on the interface
	 */
	static class CircleID {
		int ndepth;
		int nminparentlevel;
		BigInteger nscore;
		int nid;
		int nprevminparentlevel;

		CircleID(int ndepth, int nminparentlevel, int nprevminparentlevel,
				BigInteger nscore, int nid) {
			this.nminparentlevel = nminparentlevel;
			this.nprevminparentlevel = nprevminparentlevel;
			this.ndepth = ndepth;
			this.nscore = nscore;
			this.nid = nid;
		}

	}

	/**
	 * Record for the node currently selected
	 */
	static class SelectedNodeRec {
		PNode selectedNode;
		boolean bcircle;
		CircleID theCircleID = null;
	}

	/**
	 * Sets the text for the significant transcription factors
	 */
	public void setSigText(TreeSet<DREM_Timeiohmm.SigTFRecv2> tsSigTF,
			PPath boundingRect, int ntype, int ndepth, boolean displayTF,
			boolean displayMIRNA, DREM_Timeiohmm.Treenode ptr, boolean color) {
		this.displayTF = displayTF;
		this.displayMIRNA = displayMIRNA;

		// Initialize the ppath rectangles that will contain the regulators
		PPath tfUpRect = new PPath();
		PPath tfZeroRect = new PPath();
		PPath tfDownRect = new PPath();
		PPath miRNAUpRect = new PPath();
		PPath miRNAZeroRect = new PPath();
		PPath miRNADownRect = new PPath();

		tfUpRect.setPickable(false);
		tfZeroRect.setPickable(false);
		tfDownRect.setPickable(false);
		miRNAUpRect.setPickable(false);
		miRNAZeroRect.setPickable(false);
		miRNADownRect.setPickable(false);

		boundingRect.addChild(tfUpRect);
		boundingRect.addChild(tfZeroRect);
		boundingRect.addChild(tfDownRect);
		boundingRect.addChild(miRNAUpRect);
		boundingRect.addChild(miRNAZeroRect);
		boundingRect.addChild(miRNADownRect);

		float width = 0;
		float height = 0;
		float tfUpHeight = 0, tfZeroHeight = 0, tfDownHeight = 0;
		float miRNAUpHeight = 0, miRNAZeroHeight = 0, miRNADownHeight = 0;
		if (ntype == nKeyInputType) {
			HashSet<String> htAdded = new HashSet<String>();
			// First traverse the ancestors of this node to determine
			// which TFs have already appeared at previous nodes
			// on the path
			HashMap<String, Double> ancestorScores = ptr
					.getAncestorActivityScores();

			for (Object theTFRecObj : tsSigTF) {
				DREM_Timeiohmm.SigTFRecv2 theTFRec = (DREM_Timeiohmm.SigTFRecv2) theTFRecObj;
                                theTFRec.szname=theTFRec.szname.split(" ")[0]; // fix the space in the name
				Integer regType = theTimeiohmm.bindingData.regTypeMap
						.get(theTFRec.szname);
				boolean significant = false;
				int regDirection = 0;
				Integer lookup;
				/*
				 * Decide if the regulator is significant based on selected
				 * scoring method.
				 */
				if (nKeyInputType != 778 && nKeyInputType != 777) {
					switch (regType) {
					case RegulatorBindingData.MIRNA:
						if ((theTFRec.dpval <= dkeymiRNAinputpvalue)
								&& ((nKeyInputType != 1) || (theTFRec.dpercent >= dsplitpercent))
								&& (!htAdded.contains(theTFRec.szname))
								&& displayMIRNA)
							significant = true;
						break;
					case RegulatorBindingData.TF:
						if ((theTFRec.dpval <= dkeyinputpvalue)
								&& ((nKeyInputType != 1) || (theTFRec.dpercent >= dsplitpercent))
								&& (!htAdded.contains(theTFRec.szname))
								&& displayTF)
							significant = true;
						break;
					}
				} else {
					// If the activity value is above the threshold...
					// (activity values have been transformed to resemble
					// p-values here)
					if (theTFRec.dpval <= dkeyinputpvalue) {
						// ... then check if it has already appeared in a path
						// before using it
						String tfName = theTFRec.szname;

						// Only add the TF to this node's set of labels if
						// it wasn't already added to a set of labels at
						// an ancestor node on the path
						if ((nKeyInputType == 777
								|| !ancestorScores.containsKey(tfName) || ((Double) ancestorScores
								.get(tfName)).doubleValue() > dkeyinputpvalue)) {
							significant = true;
						}
					}
				}

				if (!significant)
					continue;
				
				sigRegs.add(theTFRec.szname);
				
				/*
				 * Record the expression direction to decide the color of the
				 * label's text. Blue - increased expression Black - unknown or
				 * zero expression Red - decreased expression
				 */
				switch (regType) {
				case RegulatorBindingData.MIRNA:
					/*
                                        lookup = theTimeiohmm.reg2DataSetIndex.get(theTFRec.szname
							.toUpperCase());
                                                       */
                                        
                                        lookup=Arrays.asList(theTimeiohmm.miRNADataSet.genenames).indexOf(theTFRec.szname.toUpperCase()); //fix the lookup error
                                        
					if (lookup != null) {
						// increase ndepth by 1 because of the timepoint 0 that has no exp. values
						double expression = theTimeiohmm.miRNADataSet.data[lookup][ndepth+1];
						regDirection = (int) Math.signum(expression);
					}
					break;
				case RegulatorBindingData.TF:
					lookup = theTimeiohmm.reg2DataSetIndex.get(theTFRec.szname
							.toUpperCase());
					if (lookup != null) {
						double expression = theTimeiohmm.theDataSet.data[lookup][ndepth];
						regDirection = (int) Math.signum(expression);
					}
					break;
				}

				PText text = new PText("");
				text.setFont(new Font("Arial", Font.BOLD, 14));
				text.setText(" " + theTFRec.szname + " ");
				text.setPickable(false);

				if (text.getWidth() > width)
					width = (float) text.getWidth();
				height += (float) text.getHeight();

				// Add the text to the correct rectangle and set the pen color
				// based on the direction of regulation and regulator type
				switch (regDirection) {
				case -1:
					text.setTextPaint(Color.RED);
					if (regType == RegulatorBindingData.MIRNA) {
						miRNADownRect.addChild(text);
						text.translate(0.0, miRNADownHeight);
						miRNADownHeight += (float) text.getHeight();
					} else {
						tfDownRect.addChild(text);
						text.translate(0.0, tfDownHeight);
						tfDownHeight += (float) text.getHeight();
					}
					break;
				case 0:
					if (regType == RegulatorBindingData.MIRNA) {
						miRNAZeroRect.addChild(text);
						text.translate(0.0, miRNAZeroHeight);
						miRNAZeroHeight += (float) text.getHeight();
					} else {
						tfZeroRect.addChild(text);
						text.translate(0.0, tfZeroHeight);
						tfZeroHeight += (float) text.getHeight();
					}
					break;
				case 1:
					text.setTextPaint(Color.BLUE);
					if (regType == RegulatorBindingData.MIRNA) {
						miRNAUpRect.addChild(text);
						text.translate(0.0, miRNAUpHeight);
						miRNAUpHeight += (float) text.getHeight();
					} else {
						tfUpRect.addChild(text);
						text.translate(0.0, tfUpHeight);
						tfUpHeight += (float) text.getHeight();
					}
					break;
				}
				if(!color)
					text.setTextPaint(Color.BLACK);
				htAdded.add(theTFRec.szname);
			}
		}

		boundingRect.reset();
		tfUpRect.reset();
		tfZeroRect.reset();
		tfDownRect.reset();
		miRNAUpRect.reset();
		miRNAZeroRect.reset();
		miRNADownRect.reset();

		// If any regulators were added to the rectangle draw them
		if (width != 0) {
			boundingRect.setPathToRectangle((float) 0.0, (float) 0.0, width,
					height);
			boundingRect.setPaint(Color.WHITE);

			// Draw a line from the rectangle to the point it corresponds to
			PPath line = new PPath();
			line.setPickable(false);
			boundingRect.addChild(0, line);
			if (lineendpoints.get(boundingRect) == null) {
				Point2D lineend = boundingRect
						.localToGlobal(new Point2D.Double(0, 0));
				lineendpoints.put(boundingRect, lineend);
				line.setPathToPolyline(new float[] { 0, 0 },
						new float[] { 0, 0 });
			} else {
				Point2D end = lineendpoints.get(boundingRect);
				Point2D[] linepoints = new Point2D[2];
				linepoints[0] = boundingRect.globalToLocal((Point2D) end
						.clone());
				;
				linepoints[1] = new Point2D.Double(boundingRect.getX(),
						boundingRect.getY());
				line.setPathToPolyline(linepoints);
				line.setVisible(true);
			}

			/*
			 * For each rectangle translate it to its position and then
			 * translate each of its children
			 */
			float temph = (float) 0.0;
			tfUpRect.setPathToRectangle((float) 0.0, temph, width, tfUpHeight);
			temph += tfUpHeight;
			tfZeroRect.setPathToRectangle((float) 0.0, temph, width,
					tfZeroHeight);
			for (Object o : tfZeroRect.getChildrenReference()) {
				PNode n = (PNode) o;
				n.translate(0, temph);
			}
			temph += tfZeroHeight;
			tfDownRect.setPathToRectangle((float) 0.0, temph, width,
					tfDownHeight);
			for (Object o : tfDownRect.getChildrenReference()) {
				PNode n = (PNode) o;
				n.translate(0, temph);
			}
			temph += tfDownHeight;
			miRNAUpRect.setPathToRectangle((float) 0.0, temph, width,
					miRNAUpHeight);
			for (Object o : miRNAUpRect.getChildrenReference()) {
				PNode n = (PNode) o;
				n.translate(0, temph);
			}
			temph += miRNAUpHeight;
			miRNAZeroRect.setPathToRectangle((float) 0.0, temph, width,
					miRNAZeroHeight);
			for (Object o : miRNAZeroRect.getChildrenReference()) {
				PNode n = (PNode) o;
				n.translate(0, temph);
			}
			temph += miRNAZeroHeight;
			miRNADownRect.setPathToRectangle((float) 0.0, temph, width,
					miRNADownHeight);
			for (Object o : miRNADownRect.getChildrenReference()) {
				PNode n = (PNode) o;
				n.translate(0, temph);
			}
		}
		// Possible memory leak because the old rectangles may still have
		// event listeners attached, preventing the garbage collector from
		// cleaning them up.
		// Can be observed by uncommenting the following lines and repeatedly
		// adjusting the score threshold. (Note: it may take a while to observe)
		// System.out.println("Curr memory usage: " +
		// Runtime.getRuntime().totalMemory());
	}

	/**
	 * Record for information about significant transcription factors
	 */
	public static class SigInfoRec {
		TreeSet<DREM_Timeiohmm.SigTFRecv2> theSigTF;
		int ntype;
		PNode border;
		BigInteger npathscore;
		int ndepth;
		Treenode node;

		SigInfoRec(PNode border, TreeSet<DREM_Timeiohmm.SigTFRecv2> tsSigTF,
				int ntype, BigInteger npathscore, int ndepth, Treenode node) {
			this.theSigTF = tsSigTF;
			this.ntype = ntype;
			this.border = border;
			this.npathscore = npathscore;
			this.ndepth = ndepth;
			this.node = node;
		}
	}

	static class SigTFRecv2Comp implements
			Comparator<DREM_Timeiohmm.SigTFRecv2> {
		public int compare(SigTFRecv2 arg0, SigTFRecv2 arg1) {
			if (arg0.dpval > arg1.dpval)
				return 1;
			else if (arg0.dpval < arg1.dpval)
				return -1;
			else if (arg0.dpercent > arg1.dpercent)
				return 1;
			else if (arg0.dpercent < arg1.dpercent)
				return -1;
			else
				return arg0.szname.compareTo(arg1.szname);
		}
	}

	/**
	 * Draws nodes on the interface screen
	 */
	public PBasicInputEventHandler drawNodes(DREM_Timeiohmm.Treenode ptr,
			final int ndepth, double parentx, double parenty,
			final BigInteger ncurrscore, int nchild, Color currColor,
			Color prevColor, int nprevcolorID, int ncurrparentcolorindex,
			int nprevminparentlevel) {

		PBasicInputEventHandler pbiehLine = null;

		if (ptr != null) {
			double dnodex;
			double dnodey;
			double ddiameter;

			dnodex = dwidthunitsCum[ndepth] + LEFTBUFFER;
			dnodey = REALHEIGHT - dheightunits * (ptr.dmean - dmin);
			ddiameter = Math.sqrt(dnodek) * dminheightunits
					* Math.sqrt(ptr.dsigma) / 2;
			PPath line = null;
			int nlocalcolorID = numcolor - 1;
			;

			boolean bnewcolor = true;
			if ((ptr.parent != null) && (ptr.parent.parentptrA != null)
					&& (ptr.parent.parentptrA.length != 1)) {
				currColor = prevColor;
				nlocalcolorID = nprevcolorID;
				bnewcolor = false;
			}

			if (currColor == null) {
				currColor = htColorIDtoColor.get(new Integer(numcolor));
				if (currColor == null) {
					if (numcolor < theTimeiohmm.savedColors.size()) {
						currColor = (Color) theTimeiohmm.savedColors
								.get(numcolor);
					} else {
						if (nchild == 0) {
							if (nchild < edgeColorsTriples[0].length) {
								currColor = edgeColorsTriples[nparentcolorindex
										% edgeColorsTriples.length][nchild];
							} else {
								currColor = edgeColorsTriples[(nparentcolorindex
										+ nchild - edgeColorsTriples[0].length + 1)
										% edgeColorsTriples[0].length][edgeColorsTriples[0].length - 1];
							}

							if (DREM_Timeiohmm.BDEBUG) {
								System.out.println(nparentcolorindex + "\t"
										+ nchild + "\t" + currColor);
							}

							if (ptr.parent != null) {
								ptr.parent.nparentcolorindex = nparentcolorindex;
							}
							nparentcolorindex++;
							ncurrparentcolorindex = nparentcolorindex;
						} else {
							if (nchild < edgeColorsTriples[0].length) {
								currColor = edgeColorsTriples[ptr.parent.nparentcolorindex
										% edgeColorsTriples.length][nchild];
							} else {
								currColor = edgeColorsTriples[(ptr.parent.nparentcolorindex
										+ nchild - edgeColorsTriples[0].length + 1)
										% edgeColorsTriples[0].length][edgeColorsTriples[0].length - 1];
							}

							if (DREM_Timeiohmm.BDEBUG) {
								System.out.println(ptr.parent.nparentcolorindex
										+ "\t" + nchild + "\t" + currColor);
							}
						}
					}
				}
				nlocalcolorID = numcolor;
				numcolor++;
			}

			if (bnewcolor) {
				htColorIDtoColor.put(new Integer(nlocalcolorID), currColor);
			}

			htLineIDtoColorID.put(ndepth + ";" + ncurrscore, new Integer(
					nlocalcolorID));

			PDragEventHandler borderTextDrag = new PDragEventHandler() {
				PPath border;
				Point2D zero;

				public void startDrag(PInputEvent e) {
					super.startDrag(e);
					border = (PPath) e.getPickedNode();
					zero = lineendpoints.get(border);
					e.setHandled(true);
				}

				public void drag(PInputEvent e) {
					super.drag(e);
					border = (PPath) e.getPickedNode();
					PPath line = (PPath) border.getChild(0);
					Point2D[] linepoints = new Point2D[2];
					linepoints[0] = border
							.globalToLocal((Point2D) zero.clone());
					;
					linepoints[1] = new Point2D.Double(border.getX(), border
							.getY());
					line.setPathToPolyline(linepoints);
					e.setHandled(true);
				}

				public void endDrag(PInputEvent e) {
					super.endDrag(e);
					border = (PPath) e.getPickedNode();
					PPath line = (PPath) border.getChild(0);
					Point2D[] linepoints = new Point2D[2];
					linepoints[0] = border
							.globalToLocal((Point2D) zero.clone());
					;
					linepoints[1] = new Point2D.Double(border.getX(), border
							.getY());
					line.setPathToPolyline(linepoints);
					e.setHandled(true);
				}
			};

			if (ndepth == 0) {
				final int fnchild = nchild;
				final DREM_Timeiohmm.Treenode fptr = ptr;
				final DREMGui theDREMGui = this;
				pbiehLine = new PBasicInputEventHandler() {
					public void mousePressed(PInputEvent event) {
						if (event.getButton() == MouseEvent.BUTTON3) {
							javax.swing.SwingUtilities
									.invokeLater(new Runnable() {
										public void run() {
											JFrame frame = new JFrame(
													"Path Table "
															+ nKeyInputType);
											frame
													.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
											frame.setLocation(200, 200);
                                                                                        // FIXME: edgeTable
											DREMGui_EdgeTable newContentPane = new DREMGui_EdgeTable(
													theDREMGui, frame,
													theTimeiohmm, fptr,
													ncurrscore, ndepth,
													fnchild, nKeyInputType,
													true);
											newContentPane.setOpaque(true); // content
											// panes
											// must
											// be
											// opaque
											frame
													.setContentPane(newContentPane);

											// Display the window.
											frame.pack();
											frame.setVisible(true);
										}
									});
						}
					}
				};
			} else if (ndepth > 0) {
				line = PPath.createLine((float) parentx, (float) parenty,
						(float) dnodex, (float) dnodey);
				PPath line2 = PPath.createLine((float) parentx,
						(float) parenty, (float) dnodex, (float) dnodey);

				line2.setStroke(new BasicStroke(8));
				line2.setStrokePaint(Color.black);
				canvas.getLayer().addChild(line2);

				line.setStroke(new BasicStroke(5));
				line.setStrokePaint(currColor);
				ArrayList<PPath> linesList = htColorIDtoLinesList
						.get(new Integer(nlocalcolorID));
				if (linesList == null) {
					linesList = new ArrayList<PPath>();
					linesList.add(line);
					htColorIDtoLinesList.put(new Integer(nlocalcolorID),
							linesList);
				} else {
					linesList.add(line);
				}

				if (!bglobalnode) {
					line.setVisible(false);
					line.setPickable(false);
					line2.setVisible(false);
					line2.setPickable(false);
				}
				canvas.getLayer().addChild(line);

				hideList.add(line);
				hideList.add(line2);

				final int fnchild = nchild;
				final DREM_Timeiohmm.Treenode fptr = ptr;
				final DREMGui theDREMGui = this;
				pbiehLine = new PBasicInputEventHandler() {
					public void mousePressed(PInputEvent event) {
						if (event.getButton() == MouseEvent.BUTTON3) {
							javax.swing.SwingUtilities
									.invokeLater(new Runnable() {
										public void run() {
											JFrame frame = new JFrame(
													"Path Table "
															+ nKeyInputType);
											frame
													.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
											frame.setLocation(200, 200);

											DREMGui_EdgeTable newContentPane = new DREMGui_EdgeTable(
													theDREMGui, frame,
													theTimeiohmm, fptr.parent,
													ncurrscore, ndepth,
								 					fnchild, nKeyInputType,
													false);
											newContentPane.setOpaque(true); // content
											// panes
											// must
											// be
											// opaque
											frame
													.setContentPane(newContentPane);

											// Display the window.
											frame.pack();
											frame.setVisible(true);
										}
									});
						}
					}
				};
				line.addInputEventListener(pbiehLine);
			}

			PNode circle = PPath.createEllipse(
					(float) (dnodex - ddiameter / 2.0),
					(float) (dnodey - ddiameter / 2.0), (float) ddiameter,
					(float) ddiameter);
			String szTip = "Mean " + nf3.format(ptr.dmean) + "; Std. Dev. "
					+ nf3.format(ptr.dsigma);
			ToolTipTextUpdater tipEvent = new ToolTipTextUpdater(canvas,
					circle, szTip);
			circle.addInputEventListener(tipEvent);

			if (ptr.numchildren <= 1) {
				ArrayList<PNode> circleList = htColorIDtoCircleList
						.get(new Integer(nlocalcolorID));
				if (circleList == null) {
					circleList = new ArrayList<PNode>();
					circleList.add(circle);
					htColorIDtoCircleList.put(new Integer(nlocalcolorID),
							circleList);
				} else {
					circleList.add(circle);
				}

				if ((ptr.parentptrA == null) || (ptr.parentptrA.length == 1)) {
					circle.setPaint(currColor);
				} else {
					circle.setPaint(prevColor);
				}
			} else {
				circle.setPaint(SPLITCOLOR);
			}

			PBasicInputEventHandler[] pbiehLineUse = new PBasicInputEventHandler[ptr.numchildren];

			for (int nnextchild = 0; nnextchild < ptr.numchildren; nnextchild++) {
				BigInteger npathscore = ncurrscore.add((new BigInteger(""
						+ nnextchild)).multiply((new BigInteger(""
						+ theTimeiohmm.nmaxchild))
						.pow((data[0].length - ndepth - 1))));
				// ncurrscore+nnextchild*(int)
				// Math.pow(theTimeiohmm.nmaxchild,data[0].length-ndepth-1);
				if (ptr.numchildren == 1) {
					pbiehLineUse[nnextchild] = drawNodes(
							ptr.nextptr[nnextchild], ndepth + 1, dnodex,
							dnodey, npathscore, nnextchild, currColor,
							prevColor, nprevcolorID, ncurrparentcolorindex,
							ptr.nminparentlevel);
				} else {
					pbiehLineUse[nnextchild] = drawNodes(
							ptr.nextptr[nnextchild], ndepth + 1, dnodex,
							dnodey, npathscore, nnextchild, null, currColor,
							nlocalcolorID, ncurrparentcolorindex,
							ptr.nminparentlevel);
				}
			}

			final PNode fcircle = circle;
			final DREM_Timeiohmm.Treenode fptr = ptr;
			
			if ((ptr.parentptrA == null) || (ptr.parentptrA.length == 1)) {
				htNodes.put(circle, new CircleID(ndepth, ptr.nminparentlevel,
						ptr.nminparentlevel, ncurrscore, ncircleID));
			} else {
				if (DREM_Timeiohmm.BDEBUG) {
					System.out.println(ptr.nminparentlevel + " "
							+ nprevminparentlevel + "!!!!");
				}

				htNodes.put(circle, new CircleID(ndepth, ptr.nminparentlevel,
						nprevminparentlevel, ncurrscore, ncircleID));
			}

			NumberFormat nf1 = NumberFormat.getInstance(Locale.ENGLISH);

			if (ptr.dpredictweight < 1) {
				nf1.setMaximumIntegerDigits(0);
				nf1.setMinimumFractionDigits(2);
				nf1.setMaximumFractionDigits(2);
			} else {
				nf1.setMinimumFractionDigits(1);
				nf1.setMaximumFractionDigits(1);
			}

			ptr.thepredictText = new PText(nf1.format(ptr.dpredictweight));
			ptr.thepredictText.setVisible(bshowpredict
					&& (bglobalnode || !battachlabels));
			ptr.thepredictText.translate(dnodex - ddiameter / 2.0, dnodey
					- ddiameter / 3.0);
			ptr.thepredictText.setFont(new Font("times", Font.BOLD, 10));
			ptr.thepredictText.setTextPaint(predictLabelColor);
			ptr.thepredictText.setPickable(false);

			ptr.goText = new PText(ptr.szgolabel);
			ptr.goText.setVisible(bshowgolabels && bapplygolabels
					&& (bglobalnode || !battachlabels));
			ptr.goText.setFont(new Font("times", Font.BOLD, 14));
			ptr.goText.setTextPaint(Color.black);
			ptr.goText.setPickable(false);
			ptr.goText.translate(dnodex + ddiameter / 2.0, dnodey - ddiameter
					/ 2.0);
			hidegolabelsList.add(ptr.goText);

			ptr.genesetText = new PText(ptr.szgenesetlabel);
			ptr.genesetText.setVisible(bshowgenesetlabels
					&& bapplygenesetlabels && (bglobalnode || !battachlabels));
			ptr.genesetText.setFont(new Font("times", Font.BOLD, 14));
			ptr.genesetText.setTextPaint(genesetLabelColor);
			ptr.genesetText.setPickable(false);
			ptr.genesetText.translate(dnodex + ddiameter / 2.0, dnodey
					- ddiameter / 2.0);
			hidegenesetlabelsList.add(ptr.genesetText);

			ptr.tfsetText = new PText(ptr.sztfsetlabel);
			ptr.tfsetText.setVisible(bshowtfsetlabels && bapplytfsetlabels
					&& (bglobalnode || !battachlabels));
			ptr.tfsetText.setFont(new Font("times", Font.BOLD, 14));
			ptr.tfsetText.setTextPaint(tfLabelColor);
			ptr.tfsetText.setPickable(false);
			ptr.tfsetText.translate(dnodex + ddiameter / 2.0, dnodey
					- ddiameter / 2.0);
			hidetfsetlabelsList.add(ptr.tfsetText);
			hidepredictlabelsList.add(ptr.thepredictText);

			for (int nchildindex = 0; nchildindex < ptr.numchildren; nchildindex++) {
				PPath border = new PPath();

				if ((!bglobalnode) && (battachlabels) || (!bshowkeyinputs)) {
					border.setVisible(false);
					border.setPickable(false);
				}

				double dnexty = REALHEIGHT - dheightunits
						* (ptr.nextptr[nchildindex].dmean - dmin);
				double dcurry = REALHEIGHT - dheightunits * (ptr.dmean - dmin);

				if (ptr.tsSigTFFull != null) {
					setSigText(ptr.tsSigTFFull[nchildindex], border, 3,
							ptr.ndepth, displayTF, displayMIRNA, ptr, colorSigTFs);
					border.translate(dnodex + dwidthunitsInterval[ndepth] / 2,
							(dnexty + dcurry) / 2);
					// int nextscore = ncurrscore+
					// nchildindex*(int)
					// Math.pow(theTimeiohmm.nmaxchild,data[0].length-ndepth-1);
					BigInteger nextscore = ncurrscore.add((new BigInteger(""
							+ nchildindex)).multiply((new BigInteger(""
							+ theTimeiohmm.nmaxchild)).pow((data[0].length
							- ndepth - 1))));
					hidesigList.add(new SigInfoRec(border,
							ptr.tsSigTFFull[nchildindex], 2, nextscore, ndepth,
							ptr));

					border.addInputEventListener(borderTextDrag);
					border.addInputEventListener(pbiehLineUse[nchildindex]);

					if (htTextVisible != null) {
						boolean bshow = (htTextVisible.get(ndepth + ";"
								+ nextscore + ";" + 2)).booleanValue();

						for (Object osmallrect : border.getChildrenReference()) {
							PNode smallrect = ((PNode) osmallrect);
							smallrect.setVisible(bshow);
							for (Object otext : smallrect
									.getChildrenReference()) {
								PNode text = (PNode) otext;
								text.setVisible(bshow);
							}
						}
					}
				}
			}

			PNode borderCircle;
			if (ptr.numchildren >= 2) {
				final DREMGui ftheDREMGui = this;
				PBasicInputEventHandler pbiehCircle1 = new PBasicInputEventHandler() {
					public void mousePressed(PInputEvent event) {
						if (event.getButton() == MouseEvent.BUTTON3) {
							javax.swing.SwingUtilities
									.invokeLater(new Runnable() {
										public void run() {
											JFrame frame = new JFrame(
													"Split Table");
											frame
													.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
											frame.setLocation(200, 200);
											CircleID theCircleID = htNodes
													.get(fcircle);

											if (fptr.numchildren == 2) {
												DREMGui_SplitTable newContentPane = new DREMGui_SplitTable(
														ftheDREMGui, frame,
														theTimeiohmm, fptr,
														theCircleID.nscore);
												newContentPane.setOpaque(true); // content
												// panes
												// must
												// be
												// opaque
												frame
														.setContentPane(newContentPane);
											} else {
												JTabbedPane tabbedPane = new JTabbedPane();

												for (int ntable = 0; ntable < fptr.numchildren; ntable++) {
													DREMGui_SplitTable newContentPane = new DREMGui_SplitTable(
															ftheDREMGui, frame,
															theTimeiohmm, fptr,
															theCircleID.nscore,
															ntable,
															fptr.orderA[ntable]);
													String szLabel;
													String szToolTip;
													if (fptr.numchildren == 3) {
														if (ntable == 0) {
															szLabel = "Low vs. Others ";
															szToolTip = "Low";
														} else if (ntable == 1) {
															szLabel = "Middle vs. Others ";
															szToolTip = "Middle";
														} else {
															szLabel = "High vs. Others ";
															szToolTip = "High";
														}
													} else {
														szLabel = "Child "
																+ ntable
																+ " vs. Others ";
														szToolTip = "Child "
																+ ntable;
													}
													tabbedPane.addTab(szLabel,
															null,
															newContentPane,
															szToolTip);
												}
												tabbedPane.setOpaque(true);
												frame
														.setContentPane(tabbedPane);
											}

											// Display the window.
											frame.pack();
											frame.setVisible(true);
										}
									});
						}
					}
				};
				
				borderCircle = new PPath();
				if ((!bglobalnode) && (battachlabels) || (!bshowkeyinputs)) {
					borderCircle.setVisible(false);
					borderCircle.setPickable(false);
				}

				borderCircle.translate(dnodex + ddiameter / 2, dnodey
						- ddiameter / 2);
				setSigText(ptr.tsSigTF, (PPath) borderCircle, 0, ptr.ndepth,
						displayTF, displayMIRNA, ptr, colorSigTFs);

				if (htTextVisible != null) {
					boolean bshow = (htTextVisible.get(ndepth + ";"
							+ ncurrscore + ";" + 0)).booleanValue();
					for (Object osmallrect : borderCircle
							.getChildrenReference()) {
						PNode smallrect = ((PNode) osmallrect);
						smallrect.setVisible(bshow);
						for (Object otext : smallrect.getChildrenReference()) {
							PNode text = (PNode) otext;
							text.setVisible(bshow);
						}
					}
				}
				hidesigList.add(new SigInfoRec(borderCircle, ptr.tsSigTF, 0,
						ncurrscore, ndepth, ptr));

				// Repeat for the TF activity scores

				for (int nchildindex = 0; nchildindex < ptr.numchildren; nchildindex++) {
					PNode borderCircleAct = new PPath();
					if ((!bglobalnode) && (battachlabels) || (!bshowkeyinputs)) {
						borderCircleAct.setVisible(false);
						borderCircleAct.setPickable(false);
					}
					
					double dnexty = REALHEIGHT - dheightunits
						* (ptr.nextptr[nchildindex].dmean - dmin);
					double dcurry = REALHEIGHT - dheightunits
						* (ptr.dmean - dmin);

					borderCircleAct.translate(dnodex + dwidthunitsInterval[ndepth] / 2,
							(dnexty + dcurry) / 2);
	
//					borderCircleAct.translate(dnodex + ddiameter / 2, dnodey
//							- ddiameter / 2);
					setSigText(ptr.tsSigTFActivity[nchildindex], (PPath) borderCircleAct, 777,
							ptr.ndepth, displayTF, displayMIRNA, ptr, colorSigTFs);
	
					if (htTextVisible != null) {
						boolean bshow = ((Boolean) htTextVisible.get(ndepth + ";"
								+ ncurrscore + ";" + 777)).booleanValue();
						for (Object osmallrect : borderCircleAct
								.getChildrenReference()) {
							PNode smallrect = ((PNode) osmallrect);
							smallrect.setVisible(bshow);
							for (Object otext : smallrect.getChildrenReference()) {
								PNode text = (PNode) otext;
								text.setVisible(bshow);
							}
						}
					}
					hidesigList.add(new SigInfoRec(borderCircleAct,
							ptr.tsSigTFActivity[nchildindex], 777, ncurrscore, ndepth, ptr));
	
					// Repeat for the TF activity scores that are shown
					// only the first time the TF is active on a path
					PNode borderCircleFirstAct = new PPath();
					if ((!bglobalnode) && (battachlabels) || (!bshowkeyinputs)) {
						borderCircleFirstAct.setVisible(false);
						borderCircleFirstAct.setPickable(false);
					}
	
					borderCircleFirstAct.translate(dnodex + dwidthunitsInterval[ndepth] / 2,
							(dnexty + dcurry) / 2);
					
//					borderCircleFirstAct.translate(dnodex + ddiameter / 2, dnodey
//							- ddiameter / 2);
					setSigText(ptr.tsSigTFActivity[nchildindex], (PPath) borderCircleFirstAct,
							778, ptr.ndepth, displayTF, displayMIRNA, ptr, colorSigTFs);
	
					if (htTextVisible != null) {
						boolean bshow = ((Boolean) htTextVisible.get(ndepth + ";"
								+ ncurrscore + ";" + 778)).booleanValue();
						for (Object osmallrect : borderCircleFirstAct
								.getChildrenReference()) {
							PNode smallrect = ((PNode) osmallrect);
							smallrect.setVisible(bshow);
							for (Object otext : smallrect.getChildrenReference()) {
								PNode text = (PNode) otext;
								text.setVisible(bshow);
							}
						}
					}
	
					hidesigList.add(new SigInfoRec(borderCircleFirstAct,
							ptr.tsSigTFActivity[nchildindex], 778, ncurrscore, ndepth, ptr));
					

					borderCircleAct.addInputEventListener(pbiehCircle1);
					borderCircleFirstAct.addInputEventListener(pbiehCircle1);
					borderCircleAct.addInputEventListener(borderTextDrag);
					borderCircleFirstAct.addInputEventListener(borderTextDrag);
				}
				
				
				for (int nchildindex = 0; nchildindex < ptr.numchildren; nchildindex++) {
					PNode border = new PPath();

					if ((!bglobalnode) && (battachlabels) || (!bshowkeyinputs)) {
						border.setVisible(false);
						border.setPickable(false);
					}

					setSigText(ptr.tsSigTFEdgeSplit[nchildindex],
							(PPath) border, 3, ptr.ndepth, displayTF,
							displayMIRNA, ptr, colorSigTFs);

					double dnexty = REALHEIGHT - dheightunits
							* (ptr.nextptr[nchildindex].dmean - dmin);
					double dcurry = REALHEIGHT - dheightunits
							* (ptr.dmean - dmin);

					border.translate(dnodex + dwidthunitsInterval[ndepth] / 2,
							(dnexty + dcurry) / 2);

					border.addInputEventListener(pbiehLineUse[nchildindex]);
					border.addInputEventListener(borderTextDrag);
					BigInteger nextscore = ncurrscore.add((new BigInteger(""
							+ nchildindex)).multiply((new BigInteger(""
							+ theTimeiohmm.nmaxchild)).pow((data[0].length
							- ndepth - 1))));
					// int nextscore = ncurrscore+nchildindex*(int)
					// Math.pow(theTimeiohmm.nmaxchild,data[0].length-ndepth-1);
					if (htTextVisible != null) {
						boolean bshow = (htTextVisible.get(ndepth + ";"
								+ nextscore + ";" + 1)).booleanValue();
						for (Object osmallrect : border.getChildrenReference()) {
							PNode smallrect = ((PNode) osmallrect);
							smallrect.setVisible(bshow);
							for (Object otext : smallrect
									.getChildrenReference()) {
								PNode text = (PNode) otext;
								text.setVisible(bshow);
							}
						}
					}

					hidesigList.add(new SigInfoRec(border,
							ptr.tsSigTFEdgeSplit[nchildindex], 1, nextscore,
							ndepth, ptr));
				}

				

				circle.addInputEventListener(pbiehCircle1);
				borderCircle.addInputEventListener(pbiehCircle1);
				borderCircle.addInputEventListener(borderTextDrag);
			} else {
				circle.addInputEventListener(pbiehLine);
			}

			if ((theSelectedRec.theCircleID != null)
					&& (theSelectedRec.theCircleID.nid == ncircleID)) {
				if (theSelectedRec.bcircle) {
					htColors.put(circle, (Color) circle.getPaint());
					circle.setPaint(Color.yellow);
					theSelectedRec.selectedNode = circle;
				} else {
					htColors.put(line, (Color) line.getStrokePaint());
					line.setStrokePaint(Color.yellow);
					theSelectedRec.selectedNode = line;
				}
			}

			class myPBasicInputEventHandler extends PBasicInputEventHandler {
				boolean bcircle;

				myPBasicInputEventHandler(boolean bcircle) {
					this.bcircle = bcircle;
				}

				public void mousePressed(PInputEvent event) {
					if (event.getButton() == MouseEvent.BUTTON1) {
						boolean badd = true;
						PNode pickednode = event.getPickedNode();
						if (theSelectedRec.selectedNode != null) {
							CircleID theCircleID = theSelectedRec.theCircleID;
							htRequired.remove(theCircleID);
							// removes the current circle ID from the set of
							// required circle IDs
							Color oldcolor = htColors
									.get(theSelectedRec.selectedNode);
							if (theSelectedRec.bcircle) {
								theSelectedRec.selectedNode.setPaint(oldcolor);
							} else {
								((PPath) theSelectedRec.selectedNode)
										.setStrokePaint(oldcolor);
							}

							// sets paint back to original color
							Enumeration<BigInteger> enumPaths = htHidden.keys();

							while (enumPaths.hasMoreElements()) {
								BigInteger npath = (BigInteger) enumPaths
										.nextElement();
								// int npath = intPath.intValue();

								// int nval;
								BigInteger nval = new BigInteger(""
										+ theTimeiohmm.nmaxchild);
								if (theSelectedRec.bcircle) {
									// nval = (int)
									// Math.pow(theTimeiohmm.nmaxchild,data[0].length-
									// theCircleID.nminparentlevel);
									nval = nval
											.pow((data[0].length - theCircleID.nminparentlevel));

								} else {
									// nval = (int)
									// Math.pow(theTimeiohmm.nmaxchild,data[0].length-theCircleID.nprevminparentlevel);
									nval = nval
											.pow((data[0].length - theCircleID.nprevminparentlevel));
								}

								if (DREM_Timeiohmm.BDEBUG) {
									System.out.println("B" + theCircleID.nscore
											+ " " + nval + " " + npath + " "
											+ data[0].length + " "
											+ theCircleID.nminparentlevel + " "
											+ theCircleID.nprevminparentlevel
											+ " " + nval + " " + npath + " "
											+ theCircleID.ndepth);
								}

								// if ((theCircleID.nscore/nval) !=
								// (npath/nval))
								if (!theCircleID.nscore.divide(nval).equals(
										npath.divide(nval))) {

									int nblocking = (htHidden.get(npath))
											.intValue();
									nblocking--;
									if (DREM_Timeiohmm.BDEBUG) {
										System.out
												.println("hereB " + nblocking);
									}

									htHidden.put(npath, new Integer(nblocking));
									if (nblocking == 0) {
										ArrayList<Integer> lines = htPathToLineSet
												.get(npath);
										int nsize = lines.size();
										for (int nindex = 0; nindex < nsize; nindex++) {
											int ngeneindex = ((Integer) lines
													.get(nindex)).intValue();
											bPathVisible[ngeneindex] = true;
											boolean bvisible = bglobalVisible
													&& bTFVisible[ngeneindex]
													&& bGOVisible[ngeneindex]
													&& bSetVisible[ngeneindex];
											plArray[ngeneindex]
													.setVisible(bvisible);
											plArray[ngeneindex]
													.setPickable(bvisible);
										}
									}
								}
							}

							if (pickednode.equals(theSelectedRec.selectedNode)) {
								// current node will no longer be yellow
								theSelectedRec.selectedNode = null;
								theSelectedRec.theCircleID = null;
								badd = false;
								setFilterText();
							}
						}

						if (badd) {
							CircleID theCircleID = htNodes.get(fcircle);
							htRequired.add(theCircleID);
							if (bcircle) {
								htColors.put(pickednode, (Color) pickednode
										.getPaint());
								pickednode.setPaint(Color.yellow);
							} else {
								htColors.put(pickednode,
										(Color) ((PPath) pickednode)
												.getStrokePaint());
								((PPath) pickednode)
										.setStrokePaint(Color.yellow);
							}

							theSelectedRec.selectedNode = pickednode;
							theSelectedRec.bcircle = bcircle;
							theSelectedRec.theCircleID = theCircleID;
							setFilterText();

							Enumeration<BigInteger> els = htPathToLineSet
									.keys();
							while (els.hasMoreElements()) {
								BigInteger nintPath = (BigInteger) els
										.nextElement();

								// int nintPath = intPath.intValue();

								// int nval;
								BigInteger nval = new BigInteger(""
										+ theTimeiohmm.nmaxchild);
								if (bcircle) {
									// nval = (int)
									// Math.pow(theTimeiohmm.nmaxchild,data[0].length-
									// theCircleID.ninparentlevel);
									nval = nval
											.pow((data[0].length - theCircleID.nminparentlevel));
								} else {
									// nval = (int)
									// Math.pow(theTimeiohmm.nmaxchild,data[0].length-theCircleID.nprevminparentlevel);
									nval = nval
											.pow((data[0].length - theCircleID.nprevminparentlevel));
								}

								if (DREM_Timeiohmm.BDEBUG) {
									System.out.println("A" + theCircleID.nscore
											+ " " + nval + " " + nintPath + " "
											+ data[0].length + " "
											+ theCircleID.nminparentlevel + " "
											+ theCircleID.nprevminparentlevel
											+ " " + nval + " " + nintPath);
								}

								// if ((theCircleID.nscore/nval) !=
								// (nintPath/nval))
								if (!(theCircleID.nscore.divide(nval)
										.equals(nintPath.divide(nval)))) {
									Integer intBlocking = htHidden
											.get(nintPath);
									if (DREM_Timeiohmm.BDEBUG) {
										System.out.println("hereA "
												+ intBlocking);
									}

									if ((intBlocking == null)
											|| (intBlocking.intValue() == 0)) {
										ArrayList<Integer> lines = htPathToLineSet
												.get(nintPath);
										int nsize = lines.size();
										for (int nindex = 0; nindex < nsize; nindex++) {
											int ngeneindex = ((Integer) lines
													.get(nindex)).intValue();
											bPathVisible[ngeneindex] = false;
											plArray[ngeneindex]
													.setVisible(false);
											plArray[ngeneindex]
													.setPickable(false);
										}
										htHidden.put(nintPath, new Integer(1));
									} else {
										int nBlocking = intBlocking.intValue();
										htHidden.put(nintPath, new Integer(
												nBlocking + 1));
									}
								}
							}

							if (!bholdedge) {
								if (bcircle) {
									ncolortime = Math.min(
											theCircleID.ndepth + 1,
											data[0].length - 1);
								} else {
									ncolortime = theCircleID.ndepth;
								}
								if (theYScalegui != null) {
									theYScalegui.theColorSlider
											.setValue(ncolortime);
								}

								setGeneColors();
							}
						}
					}
				}
			}
			;

			PBasicInputEventHandler pbiehc = new myPBasicInputEventHandler(true);
			circle.addInputEventListener(pbiehc);

			if (line != null) {
				PBasicInputEventHandler pbiehl = new myPBasicInputEventHandler(
						false);
				line.addInputEventListener(pbiehl);
			}
			hideList.add(circle);

			circleSet.add(new CircleRec(circle, ddiameter, ncircleID));
			ncircleID++;
		}

		return pbiehLine;
	}

	/**
	 * Record for the circle nodes on the interface
	 */
	static class CircleRec {

		double ddiameter;
		int nid;
		PNode circle;

		CircleRec(PNode circle, double ddiameter, int nid) {
			this.circle = circle;
			this.ddiameter = ddiameter;
			this.nid = nid;
		}
	}

	/**
	 * Comparator for circlerec
	 */
	static class CircleRecCompare implements Comparator<CircleRec> {
		/**
		 * Nodes with greater diameter get lower priorty, then nodes with lower
		 * ID
		 */
		public int compare(CircleRec cr1, CircleRec cr2) {
			if (cr1.ddiameter > cr2.ddiameter)
				return -1;
			else if (cr1.ddiameter < cr2.ddiameter)
				return 1;
			else if (cr1.nid < cr2.nid)
				return -1;
			else if (cr1.nid > cr2.nid)
				return 1;
			else
				return 0;

		}
	}

	// ////////////////////////////////////////////////////////////////////////////////
	/**
	 * Writes the significant regulators to a file
	 */
	public void printSigRegs(PrintWriter pwregs) {
		pwregs.println("This file contains all the significant regulators displayed on "
				+ "the model in no particular order.");
		pwregs.println("Significant regulators: ");
		for(String reg : sigRegs)
		{
			pwregs.println(reg);
		}
	}

	/**
	 * Writes the default settings to a file
	 */
	public void printDefaults(PrintWriter pwsave) {

		pwsave.println("#Main Input:");
		pwsave.println("TF-gene_Interaction_Source" + "\t"
				+ theTimeiohmm.szstaticsourceval);
		pwsave.println("TF-gene_Interactions_File" + "\t"
				+ theTimeiohmm.szbinding);
		pwsave.println("Expression_Data_File" + "\t"
				+ theTimeiohmm.theDataSet.szInputFile);
		pwsave.println("Saved_Model_File" + "\t" + theTimeiohmm.szinitfileval);
		pwsave.println("Gene_Annotation_Source" + "\t"
				+ theTimeiohmm.theDataSet.tga.szorganismsourceval);
		pwsave.println("Gene_Annotation_File" + "\t"
				+ theTimeiohmm.theDataSet.tga.szGoFile);
		pwsave.println("Cross_Reference_Source" + "\t"
				+ theTimeiohmm.theDataSet.tga.szxrefsourceval);
		pwsave.println("Cross_Reference_File" + "\t"
				+ theTimeiohmm.theDataSet.tga.szxrefval);

		pwsave
				.print("Normalize_Data[Log normalize data,Normalize data,No normalization/add 0]"
						+ "\t");
		if (theTimeiohmm.theDataSet.btakelog) {
			pwsave.println("Log normalize data");
		} else if (theTimeiohmm.theDataSet.badd0) {
			pwsave.println("No normalization/add 0");
		} else {
			pwsave.println("Normalize data");
		}
		pwsave.println("Spot_IDs_included_in_the_data_file" + "\t"
				+ theTimeiohmm.theDataSet.bspotincluded);
		pwsave.println();

		// ----------------------------------------------------------------------------------

		pwsave.println("#Repeat data");
		pwsave.print("Repeat_Data_Files(comma delimited list)" + "\t");
		if (theTimeiohmm.theDataSet.otherInputFiles != null) {
			for (int nindex = 0; nindex < theTimeiohmm.theDataSet.otherInputFiles.length; nindex++) {
				pwsave.print(theTimeiohmm.theDataSet.otherInputFiles[nindex]);
				if (nindex < theTimeiohmm.theDataSet.otherInputFiles.length - 1) {
					pwsave.print(",");
				}
			}
		}
		pwsave.println();

		pwsave
				.print("Repeat_Data_is_from[Different time periods,The same time period]"
						+ "\t");
		if (theTimeiohmm.theDataSet.bfullrepeat) {
			pwsave.println("Different time periods");
		} else {
			pwsave.println("The same time period");
		}

		pwsave.println();

		// ----------------------------------------------------------------------------------

		pwsave.println("#Filtering:");
		pwsave.println("Filter_Gene_If_It_Has_No_Static_Input_Data\t"
				+ theTimeiohmm.bfilterbinding);
		pwsave.println("Maximum_Number_of_Missing_Values" + "\t"
				+ theTimeiohmm.theDataSet.nmaxmissing);
		pwsave.println("Minimum_Correlation_between_Repeats" + "\t"
				+ theTimeiohmm.theDataSet.dmincorrelation);
		pwsave.println("Minimum_Absolute_Log_Ratio_Expression" + "\t"
				+ theTimeiohmm.theDataSet.dthresholdvalue);
		pwsave
				.print("Change_should_be_based_on[Maximum-Minimum,Difference From 0]"
						+ "\t");
		if (theTimeiohmm.theDataSet.bmaxminval) {
			pwsave.println("Maximum-Minimum");
		} else {
			pwsave.println("Difference From 0");
		}
		pwsave.println("Pre-filtered_Gene_File" + "\t"
				+ theTimeiohmm.theDataSet.tga.szextraval);
		pwsave.println();

		// ----------------------------------------------------------------------------------

		pwsave.println("#Search Options");
		pwsave.println("Allow_Path_Merges\t" + theTimeiohmm.ballowmergeval);
		pwsave.println("Maximum_number_of_paths_out_of_split\t"
				+ theTimeiohmm.nmaxchild);
		pwsave
				.println("Use_transcription_factor-gene_interaction_data_to_build\t"
						+ (!theTimeiohmm.BREGDREM));
		pwsave.print("Saved_Model[Use As Is/Start Search From/Do Not Use]\t");
		if (theTimeiohmm.ninitsearchval == 0) {
			pwsave.println("Use As Is");
		} else if (theTimeiohmm.ninitsearchval == 1) {
			pwsave.println("Start Search From");
		} else {
			pwsave.println("Do Not Use");
		}

		pwsave.println("Convergence_Likelihood_%\t"
				+ theTimeiohmm.szconvergenceval);
		pwsave
				.println("Minimum_Standard_Deviation\t"
						+ theTimeiohmm.dminstddev);
		pwsave.println();

		pwsave.println("#Model Selection Options");
		pwsave
				.println("Model_Selection_Framework[Penalized Likelihood,Train-Test]\tPenalized Likelihood");
		pwsave.println("Penalized_Likelihood_Node_Penalty\t"
				+ theTimeiohmm.sznodepenaltyval);
		pwsave.println("Random_Seed\t" + theTimeiohmm.nrandomseed);
		pwsave.println("Main_search_score_%\t" + (theTimeiohmm.EPSILON * 100));
		pwsave.println("Main_search_difference_threshold\t"
				+ theTimeiohmm.EPSILONDIFF);
		pwsave.println("Delete_path_score_%\t"
				+ (theTimeiohmm.RESPLITMIN * 100));
		pwsave.println("Delete_path_difference_threshold\t"
				+ theTimeiohmm.RESPLITMINDIFF);
		pwsave.println("Delay_split_score_%\t" + (theTimeiohmm.DELAYMIN * 100));
		pwsave.println("Delay_split_difference_threshold\t"
				+ theTimeiohmm.DELAYMINDIFF);
		pwsave.println("Merge_path_score_%\t" + (theTimeiohmm.MERGEMIN * 100));
		pwsave.println("Merge_path_difference_threshold\t"
				+ theTimeiohmm.MERGEMINDIFF);
		pwsave.println();

		// ---------------------------------------------------------------------------------
		pwsave.println("#Gene Annotations:");
		pwsave.println("Include_Biological_Process" + "\t"
				+ theTimeiohmm.theDataSet.tga.bpontoval);
		pwsave.println("Include_Molecular_Function" + "\t"
				+ theTimeiohmm.theDataSet.tga.bfontoval);
		pwsave.println("Include_Cellular_Process" + "\t"
				+ theTimeiohmm.theDataSet.tga.bcontoval);
		pwsave.println("Only_include_annotations_with_these_evidence_codes"
				+ "\t" + theTimeiohmm.theDataSet.tga.szevidenceval);
		pwsave.println("Only_include_annotations_with_these_taxon_IDs" + "\t"
				+ theTimeiohmm.theDataSet.tga.sztaxonval);
		pwsave.println("Category_ID_File" + "\t"
				+ theTimeiohmm.theDataSet.tga.szcategoryIDval);
		pwsave.println();

		// ----------------------------------------------------------------------------------
		pwsave.println("#GO Analysis:");
		pwsave
				.print("Multiple_hypothesis_correction_method_enrichment[Bonferroni,Randomization]"
						+ "\t");
		if (theTimeiohmm.theDataSet.tga.brandomgoval) {
			pwsave.println("Randomization");
		} else {
			pwsave.println("Bonferroni");
		}
		pwsave.println("Minimum_GO_level" + "\t"
				+ theTimeiohmm.theDataSet.tga.nmingolevel);
		pwsave.println("GO_Minimum_number_of_genes" + "\t"
				+ theTimeiohmm.theDataSet.tga.nmingo);
		pwsave
				.println("Number_of_samples_for_randomized_multiple_hypothesis_correction"
						+ "\t" + theTimeiohmm.theDataSet.tga.nsamplespval);
		pwsave.println();

		// ----------------------------------------------------------------------------------

		// Optionally print DECOD options if it was used
		if(theTimeiohmm.decodPath != null && !theTimeiohmm.decodPath.equals(""))
		{
			pwsave.println("#DECOD Options");
			pwsave.println("Gene_To_Fasta_Format_file\t" + theTimeiohmm.fastaDataFile);
			pwsave.println("DECOD_Executable_Path\t" + theTimeiohmm.decodPath);
			pwsave.println();
		}
		
		
		pwsave.println("#Expression Scaling Options");
		pwsave.print("Regulator_Types_Used_For_Activity_Scoring\t");
		if (theTimeiohmm.scaleMIRNAExp && theTimeiohmm.scaleTFExp)
			pwsave.println("Both");
		else if (theTimeiohmm.scaleMIRNAExp)
			pwsave.println("miRNA");
		else if (theTimeiohmm.scaleTFExp)
			pwsave.println("TF");
		else
			pwsave.println("None");
		pwsave.println("Expression_Scaling_Weight\t"
				+ theTimeiohmm.expressionScalingFactor);
		pwsave.println("Minimum_TF_Expression_After_Scaling\t"
				+ theTimeiohmm.minTFExpAfterScaling);
		pwsave.println();

		if (theTimeiohmm.miRNAInteractionDataFile != null
				&& !theTimeiohmm.miRNAInteractionDataFile.equals("")) {
			pwsave.println("#DREM with miRNA Options");
			pwsave.println("miRNA-gene_Interaction_Source\t"
					+ theTimeiohmm.miRNAInteractionDataFile);
			pwsave.println("miRNA_Expression_Data_File\t"
					+ theTimeiohmm.miRNADataSet.szInputFile);
			pwsave
					.print("Normalize_miRNA_Data[Log normalize data,Normalize data,No normalization/add 0]\t");
			if (theTimeiohmm.miRNADataSet.badd0)
				pwsave.println("No normalization/add 0");
			else if (theTimeiohmm.miRNADataSet.btakelog)
				pwsave.println("Log normalize data");
			else
				pwsave.println("Normalize data");
			pwsave.print("Repeat_miRNA_Data_Files(comma delimited list)\t");
			if (theTimeiohmm.miRNADataSet.otherInputFiles != null) {
				for (int nindex = 0; nindex < theTimeiohmm.miRNADataSet.otherInputFiles.length; nindex++) {
					pwsave
							.print(theTimeiohmm.miRNADataSet.otherInputFiles[nindex]);
					if (nindex < theTimeiohmm.miRNADataSet.otherInputFiles.length - 1) {
						pwsave.print(",");
					}
				}
			}
			pwsave.println();
			pwsave
					.print("Repeat_miRNA_Data_is_from[Different time periods,The same time period]\t");
			if (theTimeiohmm.miRNADataSet.bfullrepeat)
				pwsave.println("Different time periods");
			else
				pwsave.println("The same time period");
			pwsave.println("Filter_miRNA_With_No_Expression_Data_From_Regulators\t"
					+ theTimeiohmm.filterMIRNAExp);
			pwsave.println();
		}

		if(theTimeiohmm.regPriorsFile != null && !theTimeiohmm.regPriorsFile.equals(""))
		{
			pwsave.println("#SDREM Options");
			pwsave.println("Regulator_Score_File\t" + theTimeiohmm.regPriorsFile);
			pwsave.println("Active_TF_influence\t" + theTimeiohmm.dProbBindingFunctional);
			pwsave.println();
		}
		
		pwsave.println("#Interface Options");
		pwsave.println("X-axis_Scale_Factor\t" + dscalex);
		pwsave.println("Y-axis_Scale_Factor\t" + dscaley);
		pwsave.print("X-axis_scale_should_be[Uniform,Based on real time]"
				+ "\t");
		if (brealXaxis) {
			pwsave.println("Based on real time");
		} else {
			pwsave.println("Uniform");
		}
		NumberFormat nfpval = NumberFormat.getInstance(Locale.ENGLISH);
		nfpval.setMinimumFractionDigits(1);
		nfpval.setMaximumFractionDigits(1);
		pwsave.println("Key_Input_X_p-val_10^-X\t"
				+ (nfpval.format(-Math.log(dkeyinputpvalue) / Math.log(10))));
		pwsave.println("Minimum_Split_Percent\t" + (dsplitpercent * 100));
		pwsave.println("Scale_Node_Areas_By_The_Factor\t" + dnodek);
		pwsave
				.print("Key_Input_Significance_Based_On[Path Significance Conditional on Split,Path Significance Overall,Split Significance]\t");
		if (nKeyInputType == 0) {
			pwsave.println("Split Significance");
		} else if (nKeyInputType == 1) {
			pwsave.println("Path Significance Conditional on Split");
		} else {
			pwsave.println("Path Significance Overall");
		}
		pwsave.println();
		
		pwsave.println();
		pwsave.close();

	}
	// TODO more permanent solution
	/**
	 * Save the model and TF activities when running in non-interactive batch
	 * mode
	 * 
	 * @param filename
	 *            use this name to create the output files
	 */
	public void batchSave(String filename) {
		if (saveModelFrame == null) {
			saveModelFrame = new JFrame("Save Model to File");
			saveModelFrame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
			saveModelFrame.setLocation(400, 300);
			// Use "this" here, which is not exactly what is done in drawmain
			DREMGui_SaveModel newContentPane = new DREMGui_SaveModel(
					this.theTimeiohmm, treecopy, saveModelFrame, this);
			// newContentPane.setOpaque(true);
			// content panes must be opaque
			saveModelFrame.setContentPane(newContentPane);
			// Display the window.
			// saveModelFrame.pack();
		}

		Container c = saveModelFrame.getContentPane();
		// The content pane shoul be a DREMGUI_SaveModel
		((DREMGui_SaveModel) c).batchSave(filename);
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
        // get timePoints from input expression file
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
          
          if (theTimeiohmm.theDataSet.badd0){
              String[] TT=new String[TimePoints.length+1];
              TT[0]="0";
              for (int i=0;i<TimePoints.length;i++){
                  TT[i+1]=TimePoints[i];
              }
              return TT;
          }
          return TimePoints;
      }
        **/
        
        
      // copy folder
        public static void copyFolder(File src, File dest) throws IOException{
            if (src.isDirectory()){
                if (!dest.exists()){
                    dest.mkdir();
                    
                }
                
                String files[]=src.list();
                
                for (String file:files){
                    File srcFile=new File(src,file);
                    File destFile=new File(dest,file);
                    copyFolder(srcFile,destFile);
                }
            }else{
                    InputStream in=new FileInputStream(src);
                    OutputStream out=new FileOutputStream(dest);
                    byte[] buffer=new byte[1024];

                    int length;

                    while ((length=in.read(buffer))>0){
                        out.write(buffer,0,length);
                    }
                    
                    in.close();
                    out.close();
            }
        }
                
}

class JsonNode{
    public int parent;
    public ArrayList<Integer> children;
    public int nodeID;
    public String nodetime;
    public double nodeMean;
    public double nodeSigma;
    public boolean[] genesInNode;
    public String[][] ETF;
    public ArrayList<String[]> goNode;

    JsonNode(int nparent, ArrayList<Integer> nchildren, int nID, String nTime, double nMean, double nSigma, boolean[] inNode, String[][] tabdata,ArrayList<String[]> nodeGO) {
        parent=nparent;
        children=nchildren;
        nodeID=nID;
        nodetime=nTime;
        nodeMean=nMean;
        nodeSigma=nSigma;
        genesInNode=inNode;
        ETF=tabdata;
        goNode=nodeGO;
    }
    
    JsonNode(){
    
    }
}