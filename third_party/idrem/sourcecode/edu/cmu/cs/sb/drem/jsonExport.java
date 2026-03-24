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
public class jsonExport{              
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
            
            
            int[][] regTarget=theTimeiohmm.bindingData.gene2RegBindingIndex[0];
            ArrayList<String[]> geneAbList=new ArrayList<String[]>();
            ArrayList<String[]> mirAbList=new ArrayList<String[]>();
            ArrayList<String[]> proteAbList=new ArrayList<String[]>();
            geneAbList=getGeneAbVal(theTimeiohmm.theDataSet.szInputFile);
            
            String mirExpressionString,mirExpressionIDString;
            if (theTimeiohmm.miRNADataSet!=null){
                mirAbList=getGeneAbVal(theTimeiohmm.miRNADataSet.szInputFile);
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
            String combinedJson='['+jsonString+','+silGeneScoreString+','+regTargetString+','+geneNameString+','+regNameString+
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
                //Desktop.getDesktop().open(new File(destFolderString+"/idrem_result.html"));
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
