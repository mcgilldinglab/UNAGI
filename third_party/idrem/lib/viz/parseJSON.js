//google.charts.load('current', {packages: ['corechart']});
//google.charts.setOnLoadCallback(onload);

// wait for page to load
window.addEventListener('load', loadGoogle, false);

function loadGoogle() {
  // define callback in load statement
  google.charts.load('current', {packages: ['corechart']});
  google.charts.setOnLoadCallback(onload);
}

function onload(){
	
	  // load google visualization APIs-->
	for (var i in data){
		if (data[i]==null){
			data[i]=[];
		}
	}
	[nodes,methyscore,geneRegMap,exgene,reg,genex,mirex,mirID,geneabex,mirabex,protex]=data;	
	nodes=getNodeGenes(nodes);
	nodes=buildTree(nodes);
	nodes=sortNodeChildren(nodes);
	timePoints=getAllTime(nodes);
	regTargetMap=buildTargetMap(geneRegMap); //reg-> gene targets
	root=nodes[0];
	drawTree(root);
	createDropDown("ToppGene","#shiftclickdropdowndiv","shiftclickdropdown", ["PANTHER"]);
	createDropDown("Regulator","#clickdropdowndiv", "clickdropdown",
					["Genes Assigned To The Node","Compare Regulator",
					"Average Methylation For Genes In Node",
					"Average Methylation For All Top Regulator Targets",
					"Average Methylation For Top Regulator Targets In Node",
					"Single Cells",
					"Sorted Cells"
					]);
	createDropDown("Choose Regulator","#tfdropdown","tfdropdown",reg,tfdropdownonchange);
	createDropDown("Choose Time","#methydropdownTimediv","methytimedropdown",Object.keys(methyscore),methyTimedropdownonchange);
	
	//
	//diffmethy dropdown
	createDropDown("Choose Methylation Time 1","#diffmethydropdownTimediv1","diffmethytimedropdown1",Object.keys(methyscore));
	createDropDown("Choose Methylation Time 2","#diffmethydropdownTimediv2","diffmethytimedropdown2",Object.keys(methyscore),explorediffmethy);
	
	//regulatormethydropdown
	createDropDown("All","#methydropdownRegulatorNodediv","regulatormethynodedropdown",nodes.map(function(d){return d.nodeID;}));
	
	//show page after loading
	showPage();
	
	// cell types dropdown
	[singleCells,sortedCells]=data_cells;
	
	var singlecelltypes=[];
	for (var row of singleCells){
		var ctype=row[1];
		if (singlecelltypes.indexOf(ctype)==-1){
			singlecelltypes.push(ctype);
		}
	}
	createDropDown("Choose Single Cell Types","#singlecelldropdowndiv","singlecelldropdown",singlecelltypes,singlecelldropdownonchange)
	
	var sortedcelltypes=[];
	for (var row of sortedCells){
		var ctype=row[1];
		if (sortedcelltypes.indexOf(ctype)==-1){
			sortedcelltypes.push(ctype);
		}
	}
	
	createDropDown("Choose Sorted Cell Types","#sortedcelldropdowndiv","sortedcelldropdown",sortedcelltypes,sortedcelldropdownonchange)
		
	//compare 
	compareNodes=cdata;
	

};

//showpage after loading
var showPage=function(){
	document.getElementById("loadercontainer").style.display="none";
	document.getElementById("myviz").style.display="block";
}

/* pre-process the data*/

//get all time points
var getAllTime=function(nodes){
	var timeList=[];
	for (var node of nodes){
		if (timeList.indexOf(node.nodetime)==-1){
			timeList.push(node.nodetime);
		}
	}
	return timeList;
}

// get genes for each node
var getNodeGenes=function(nodes){
	
	for (var node of nodes){
		var gList=[];
		var bin=node.genesInNode;
		for (var i in bin){
			if (bin[i]){
				gList.push(exgene[i]);
			}
		}
		node["genes"]=gList;	
	}
	return nodes;
};

// build target map 
var buildTargetMap=function(geneRegMap){
	var regTargetMap={};
	for (var i in geneRegMap){
		var gene=exgene[i];
		for (var gt of geneRegMap[i]){
			var gtname=reg[gt].toUpperCase();
			if (!(gtname in regTargetMap)){
				regTargetMap[gtname]=[gene];
			}else{
				regTargetMap[gtname].push(gene);
			}
		}
	}
	return regTargetMap;
}

//sort children nodes based on mean expression value
var sortNodeChildren=function(nodes){
	for (var node of nodes){
		node.children.sort(nodeCompare);
	}
	return nodes;
};


//create tree structure
var buildTree=function(nodes){
	var xc;
	for (var x of nodes){
		if(x.parent!="-1"){
			x.parent=nodes[x.parent];
		}else{
			x.parent=null;
		}
		
		xc=[];
		for (var y of x.children){
			if (y!=-1){
				xc.push(nodes[y]);
			}
		}
		if(xc==[]){
			xc=null;
		}
		
		x.children=xc;
	}
	
	return nodes;	
};


//drawTree
var drawTree=function(root){
		var margin = {top: 20, right: 20, bottom: 20, left: 40},
		width = 1000 - margin.right - margin.left,
		height =1200 - margin.top - margin.bottom;
		var i = 0;
		var tree = d3.layout.tree();
		tree.size([width,height]);
		
		var diagonal = d3.svg.diagonal()
			.projection(function(d) { return [d.y, d.x]; });
		
		//default bgcolor
		var default_bgcolor="#333";
		var bgcolor;
		var input_bgcolor=d3.select("#bgcolor").value;
		colorisOK  = /(^#[0-9A-F]{6}$)|(^#[0-9A-F]{3}$)/i.test(input_bgcolor)
		if (colorisOK){
			bgcolor=input_bgcolor;
		}else{
			bgcolor=default_bgcolor;
		}
		svg = d3.select("#div_svg").append("svg")
			.attr("width", width + margin.right + margin.left)
			.attr("height", height + margin.top + margin.bottom)
			.attr("id","svg")
			.style("background",bgcolor)
			.attr("viewBox","0 0 1400 1200")
			.attr("preserveAspectRatio", "none")
			.append("g")
			.attr("transform", "translate(" + margin.left + "," + margin.top + ")");
			
		var allnodes=tree.nodes(root);
		var links=tree.links(allnodes);
		 
		// Declare the nodes
		var node = svg.selectAll("g.node")
		  .data(nodes)
		  .data(nodes, function(d) { return d.id = ++i; });

		// Enter the nodes.
		var nodeEnter = node.enter().append("g")
		  .attr("class", "node")
		  .style("font","8px Arial")
		  .attr("transform", function(d) { 	 
			  return "translate(" + d.y + "," + d.x + ")"; });
		
		//TODO: Handle click events
		nodeEnter.append("circle")
		  .attr("r", 16)
		  .attr("fill","#fff")
		  .attr("stroke","steelblue")
		  .attr("stroke-width","3px")
		  .on("mouseover",inNode)
		  .on("mouseout",outNode)
		  .on("click",function(){
				if(d3.event.shiftKey){
					//shift click
					var shiftclickdropdownValue=document.getElementById("shiftclickdropdown").value;
					if (shiftclickdropdownValue==="ToppGene"){
						toppgenego(this.__data__);
					}else{
						panthergo(this.__data__);
					}
				}else{
					// click
					var clickdropdownValue=document.getElementById("clickdropdown").value;
					switch(clickdropdownValue){
						case "Genes Assigned To The Node":
							showGenesNode(this.__data__);
							break;
						case "Average Methylation For Genes In Node":
							showMethyGenes(this.__data__.genes,'node',this.__data__.nodeID);
							break;
						case "Average Methylation For All Top Regulator Targets":
							showTopTFMethyAll(this.__data__);
							break;
						case "Average Methylation For Top Regulator Targets In Node":
							showTopTFMethy(this.__data__);
							break;
						case "Methylation Statistics":
							showMethylationStatistics(this.__data__);
							break;
						case "Compare Regulator":
							compareTF(this.__data__);
							break;
						case "Single Cells":
							showSingleCell(this.__data__);
							break;
						case "Sorted Cells":
							showSortedCell(this.__data__);
							break;
						default:
							showmouseover(this.__data__);
					}	
				}
			  });
			  
		var textEnter=nodeEnter.append("text")
		  .text(function(d){return d.nodeID+'('+d.nodetime+')';})
		  .attr("x",-20)
		  .attr("dy", "3.25em")
		  .attr("class","nodetext")
		  .attr("fill","#fff")
		  .attr("text-anchor", function(d) { 
			  return  "start"; })
		  .style("fill-opacity", 1)
		  .style("font","10px Arial");

		
		
		// Declare the links
		var link = svg.selectAll("path.link")
		  .data(links, function(d) { return d.target.id; })
		  .enter();

		// Enter the links.
		link.insert("path", "g")
		  .attr("class", "link")
		  .attr("stroke","#aaa")
		  .attr("stroke-width","2px")
		  .attr("fill","none")
		  .attr("d", diagonal)
		  .attr("id",function(d,i){return "s"+i;})
		  .attr("color","black");
		 
		 
		 //link text
		link.append("text")
		.append("textPath")
		.attr("class","linktext")
		.attr("xlink:href",function(d,i){return "#s"+i;})
		.attr("startOffset","20%")
		.attr("dy","-1em");
		 
		 // Define the div for the tooltip (Enable mouse over popup)
		tooltipdiv = d3.select("body").append("div")	
			.attr("class", "tooltip");
			
		//label path
		labelPath();
};

//label path
var labelPath=function(){
	var nodes=d3.selectAll("g.node")[0];
	var leafnodes=[];
	for (var node of nodes){
		if (node.__data__.children==undefined ||node.__data__.children.length==0){
			leafnodes.push(node);
		}
	}
	var leafMU=[];
	for (var node of leafnodes){
		leafMU.push(node.__data__.nodeMean);
	}
	
	leafMU.sort(function(a,b){return b-a});

	d3.selectAll(leafnodes)
	.append('text')
	.attr("x",30)
	.text(function(d,i){
		    var di=d.nodeMean;
		    var di=leafMU.indexOf(di);
			var chr=String.fromCharCode(65+di);
			d["path"]=chr;
			return 'Path:'+chr;
		})
	.style("font-size","12px")
	.attr("fill","#fff");
	
	leafnodesData=leafnodes.map(function(d){return d.__data__});
	
	for (var node of leafnodesData){
		var path=[];
		while(node!=undefined){
			path.push(node);
			node=node.parent;
		}
		
		for (var i=1;i<path.length;i++){
			var node=path[i];
			if ("path" in node){
				node["path"]+=","+path[0].path;;
			}else{
				node["path"]=path[0].path;;
			}
		}
	}
}


/* config set section*/

//popup change
var popupchange=function(newValue){
	document.getElementById("popupcut").innerHTML=newValue;
}

//tfCutoff selector action
var tfcutChange=function(newValue){
	document.getElementById("tfcutSelector").innerHTML=newValue;
	exploretf();
}

//zoom slider bar action
var zoom=function(newValue){
		document.getElementById("zoomslider").innerHTML=newValue;
		var wd=1000;
		var ht=1200;
		var sv=50;

		var zx=newValue/sv;
		var newwd=wd*zx;
		var newht=ht*zx;
		
		d3.select("#div_svg").select("svg")
		.attr("width",newwd)
		.attr("height",newht)
		.attr("preserveAspectRatio", "none");
};

//set go sliders
var setgoslider=function(newValue){
	document.getElementById("goslider").innerHTML=newValue;
}

var setsankeytfslider=function(newValue){
	document.getElementById("sankeytfslider").innerHTML=newValue;
}

var setsankeymirslider=function(newValue){
	document.getElementById("sankeymirslider").innerHTML=newValue;
}

////////////////////////
//set bgcolor
var setbgcolor=function(){
		var color=document.getElementById("bgcolor").value;
		d3.select("svg").style("background", color);
};

//set nodecolor
var setnodecolor=function(){
	var color=document.getElementById("nodecolor").value;
	d3.selectAll("g.node")
	.selectAll("circle")
	.attr("fill",color);
}

//set path color
var setpathcolor=function(){
	var color=document.getElementById("pathcolor").value;
	d3.select("svg")
	.selectAll("path")
	.attr("stroke",color);
}

var settextcolor=function(){
	var color=document.getElementById("textcolor").value;
	d3.selectAll("g.node")
	.selectAll("text")
	.attr("fill",color);
}

var resetnodesize=function(){
	var nodesize=document.getElementById("nodesizetext").value;
	d3.selectAll("circle")
	.attr("r",nodesize);
}

var resettextsize=function(){
	var textsize=document.getElementById("textsize").value;
	d3.selectAll("g.node")
	.selectAll("text")
	.style("font-size",textsize);
}

//reset config
var resetconfig=function(){
		zoom(50);
		tfcutChange(50);
		popupchange(20);
		setgoslider(5);
		setsankeytfslider(10);
		setsankeymirslider(5);
		
		//reset sliders
		document.getElementById("popupcutslider").value=20;
		document.getElementById('zoomsliderbar').value=50;
		document.getElementById("tfcutSelectorbar").value=50;
		document.getElementById("gosliderbar").value=5;
		document.getElementById("sankeytfsliderbar").value=10;
		document.getElementById("sankeymirsliderbar").value=5;
		
		
		document.getElementById("bgcolor").value="#333333";
		document.getElementById("nodecolor").value="#ffffff";
		document.getElementById("textcolor").value="#ffffff";
		document.getElementById("pathcolor").value="#aaaaaa";
		resetPath();
		setbgcolor();
		resetNode();
};

//reset path

var resetPath=function(){
	d3.select("svg").selectAll("path")
		.attr("stroke","#aaa");
		
	d3.selectAll(".linktext")
		.text("");
};

//reset node

var resetNode=function(){
	d3.selectAll(".nodeExText")
		.text("");
	d3.selectAll("g.node")
	.selectAll("circle")
	.attr("fill","#fff");
	d3.selectAll("g.node")
	.selectAll("text")
	.attr("fill","#fff");
};


/*node action section */

// in node function	
var inNode=function(){
	if (d3.select("#tooltipcheck").property("checked")==false){
			return false;
	}
	var showcut=20;
	showcut=document.getElementById("popupcut").innerHTML;
	showcut=parseInt(showcut);
	var etf=JSON.parse(JSON.stringify(this.__data__.ETF));
	var row=etf[0];
	var rowT1="TF	Num Total	Num Parent	Num Path	Expect Overall	Diff. Overall	Score Overall	Expect Split	Diff. Split	Score Split	% Split";
	var rowT2="TF	Num Total	Num Path	Expect Overall	Diff. Overall	Score Overall";
	var cetf=[];
	if (row.length==rowT1.split("\t").length){
		cetf.push(rowT1.split("\t"));
	}
	
	if (row.length==rowT2.split("\t").length){
		cetf.push(rowT2.split("\t"));
	}
	
	for (var i=0;i<showcut;i++){
		cetf[i][0]=cetf[i][0].split(" ")[0];
		cetf.push(etf[i]);
	}
	
	tooltipdiv
			//.style("left",(d3.event.pageX+30)+"px")
			//.style("top",(d3.event.pageY-28)+"px")
			.style("left",1200+"px")
			.style("top", 10+"px")
			.style("position","absolute")
			.style("opacity",1)
			.style("background","silver")
			.style("border","1px solid")
			.attr("class","tooltipsvg")
			.attr("width",300)
			.attr("height",300);
			
	createTFTable(tooltipdiv,"tftable",cetf,exploregene,this.__data__.depth);
};

//mouse out node function
var outNode=function(){
	
		tooltipdiv
		//.style("opacity", 0)
		.attr("width",0)
		.attr("height",0)
		.selectAll("*").remove();;
};

//show TFs in node
var compareTF=function(node){
	var showcut;
	showcut=document.getElementById("popupcut").innerHTML;
	showcut=parseInt(showcut);
	var etf=JSON.parse(JSON.stringify(node.ETF));
	var cetf=[];
	var ctf=[];
	for (var i=0;i<showcut;i++){
		cetf.push(etf[i]);
		ctf.push(etf[i][0]);
	}
	
	
	var cetf_no;
	var cetf_prote;
	var tfcompare_table=[]
	for (var cnode of compareNodes){
		var cnode_etf=[];
		var cnode_etf_no=[];
		var cnode_etf_prote=[];
		if (node.nodeID==cnode.nodeID){
		
			for (var tf of cnode.etf){
					tf=tf[0].split('\t')[0];
					cnode_etf.push(tf);
			}
			for (var tf of cnode.etf_no){
					tf=tf[0].split('\t')[0];
					cnode_etf_no.push(tf);
			}
			for (var tf of cnode.etf_prote){
					tf=tf[0].split('\t')[0];
					cnode_etf_prote.push(tf);
			}
			
			cetf_no=cnode_etf_no;
			cetf_prote=cnode_etf_prote;
			break;
		}
		
	}
	showCompare(ctf,cetf_no,cetf_prote);
};

var showMethylationStatistics=function(node){
	newWLC = open('','_blank','height=600,width=800,left=1400,top=200,scrollbars=yes');
	newWLC.document.write("<body>Impact analysis of Methylation data</body>");
	// analyze the impact of methylation 
	var showcut;
	showcut=document.getElementById("popupcut").innerHTML;
	showcut=parseInt(showcut);
	var etf=JSON.parse(JSON.stringify(node.ETF));
	var cetf=[];
	var ctf=[];
	for (var i=0;i<showcut;i++){
		cetf.push(etf[i]);
		ctf.push(etf[i][0]);
	}
	
	var r0;
	var r1;
	var r2;
	var r3;
	var r4;
	var r5;
	var r6;
	var r7;
	var r8;
	var r9;
	var TFMethyRes=[['nodeID','TF','# of TF targets','# of TF targets in parent node without methylation data','# of TF targets in parent node with methylation data','# of TF targets in node without methylation data','# of TF targets in node with methylation data','split p-value with methylation data',
	'split p-value without methylation data','split % with methylation data','split % without methylation data']];
	for (var tf of cetf){
		var tfname=tf[0].split(" ")[0];
		[r0,r1,r2,r3,r4,r5,r6,r7,r8,r9]=pvTFMethyImpact(tfname,node);
		TFMethyRes.push([r0,tfname,r1,r2,r3,r4,r5,r6,r7,r8,r9]);
	}
	
	var genediv=d3.select(newWLC.document.body)
		.style("background","white")
		.append("div")
		.style("padding-left","50px")
		.style("padding-top","50px")
		.attr("width",400)
		.attr("height",600)
		.attr("class","div_table_methy");
	createTable(genediv,"methytable",TFMethyRes,exploregene);
	//showmouseover(node);
};

//show genes assigned to the node
var showGenesNode=function(node){
	d3.event.preventDefault();
	var geneList=[];
	var geneIndex=node.genesInNode;
	for (var i in geneIndex){
		var iList=[];
		if (geneIndex[i]){
			var iex=genex[i][node.depth];
			var nodeProb=NormProb(iex,node.nodeMean,node.nodeSigma);
			var pathProb=0;
			var iNodes=getNodeList(exgene[i]);
			for (var iNode of iNodes){
				var iNodeEx=genex[i][iNode.__data__.depth];
				var iprob=NormProb(iNodeEx,iNode.__data__.nodeMean,iNode.__data__.nodeSigma)
				pathProb=pathProb+iprob;
			} 
			iList.push(exgene[i]);
			iList.push(nodeProb);
			iList.push(pathProb);
			geneList.push(iList);
		}
	}
	geneList.sort(function(a,b){
			if (a[2]<b[2]){
				return -1;
			}else if (a[2]>b[2]){
				return 1;
			}else{
				return 0;
			}
		});
	
	var geneListWithID=[['gene','score_assigned_to_node (the smaller the better)','score_assigned_to_path (the smaller the better)']];
	for (var row of geneList){
		geneListWithID.push(row);
	}
	
	
	var newW = open('','_blank','height=600,width=600,left=1400,top=200,scrollbars=yes');
	newW.document.write("<body></body>");
	var genediv=d3.select(newW.document.body)
		.style("background","white")
		.append("div")
		.style("padding-left","50px")
		.style("padding-top","50px")
		.attr("width",400)
		.attr("height",600)
		.attr("class","div_table_edge");
	createTable(genediv,"genetable",geneListWithID,exploregene);
};

//show sorted cells
var showSortedCell=function(node){
	var N=exgene.length;
	var scResultList=[]
	var firstRow=['nodeID','cell_type','#_of_overlapping_genes','overlapping_pvalue','overlapping_genes'];
	for (var cell of sortedCells){
		var cell_time=cell[0];
		var cell_type=cell[1].toUpperCase();
		var cell_target=cell[2].split(",").map(function(d){return d.toUpperCase();});
		cell_target=cell_target.filter(function(d){
				if (exgene.indexOf(d)!=-1){
					return true;
				}else{
					return false;
				}
			});
		
		if (node.nodetime==cell_time){
			var p=cell_target.length*1.0/N;
			var pv_cut=0.05;
			var timeNodes=[];
			var ov_tar=cell_target.filter(function(d){
				if (node.genes.indexOf(d)!=-1){
					return true;
				}else{
					return false;
				}
			});	
			var pv=1-pbinom(ov_tar.length-1,node.genes.length,p);
			if (pv<pv_cut){
				scResultList.push([node.nodeID,cell_type,ov_tar.length,pv,ov_tar]);
			}
		}
	}
	scResultList.sort(function(a,b){
			if (a[3]<b[3]){
				return -1;
			}else if (a[3]>b[3]){
				return 1;
			}else{
				return 0;
			}
		});
	
	//
	scResultListFR=[firstRow];
	for (var row of scResultList){
		scResultListFR.push(row);
	}
	
	var newW = open('','_blank','height=600,width=600,left=1400,top=200,scrollbars=yes');
	newW.document.write("<body></body>");
	var genediv=d3.select(newW.document.body)
		.style("background","white")
		.append("div")
		.style("padding-left","50px")
		.style("padding-top","50px")
		.style("margin-right","100px")
		.attr("width",400)
		.attr("height",600)
		.attr("class","div_table_sorted");
	createTable(genediv,"sorted_table",scResultListFR);
}


// show single cells
var showSingleCell=function(node){
	var N=exgene.length;
	var scResultList=[];
	var firstRow=['nodeID','cell_type','#_of_overlapping_genes','overlapping_pvalue','overlapping_genes'];
	for (var cell of singleCells){
		var cell_time=cell[0];
		var cell_type=cell[1].toUpperCase();
		var cell_target=cell[2].split(",").map(function(d){return d.toUpperCase();});
		cell_target=cell_target.filter(function(d){
				if (exgene.indexOf(d)!=-1){
					return true;
				}else{
					return false;
				}
			});
		if (node.nodetime==cell_time){
			var p=cell_target.length*1.0/N;
			var pv_cut=0.05;
			var timeNodes=[];
			var ov_tar=cell_target.filter(function(d){
				if (node.genes.indexOf(d)!=-1){
					return true;
				}else{
					return false;
				}
			});	
			var pv=1-pbinom(ov_tar.length-1,node.genes.length,p);
			if (pv<pv_cut){
				scResultList.push([node.nodeID,cell_type,ov_tar.length,pv,ov_tar]);
			}
		}
	}
	//
	scResultList.sort(function(a,b){
			if (a[3]<b[3]){
				return -1;
			}else if (a[3]>b[3]){
				return 1;
			}else{
				return 0;
			}
		});
	
	scResultListFR=[firstRow];
	for (var row of scResultList){
		scResultListFR.push(row);
	}
	
	var newW = open('','_blank','height=600,width=600,left=1400,top=200,scrollbars=yes');
	newW.document.write("<body></body>");
	var genediv=d3.select(newW.document.body)
		.style("background","white")
		.append("div")
		.style("padding-left","50px")
		.style("padding-top","50px")
		.style("margin-right","100px")
		.attr("width",400)
		.attr("height",600)
		.attr("class","div_table_single");
	createTable(genediv,"single_table",scResultListFR);
}



//panther analysis
var panthergo=function(node){
	
	var geneList=[];
	//var geneIndex=this.__data__.genesInNode;
	var geneIndex=node.genesInNode;
	for (var i in geneIndex){
		var iList=[];
		if (geneIndex[i]){
			var iex=genex[i][node.depth];
			var nodeProb=NormProb(iex,node.nodeMean,node.nodeSigma);
			var pathProb=0;
			var iNodes=getNodeList(exgene[i]);
			for (var iNode of iNodes){
				var iNodeEx=genex[i][iNode.__data__.depth];
				var iprob=NormProb(iNodeEx,iNode.__data__.nodeMean,iNode.__data__.nodeSigma)
				pathProb=pathProb+iprob;
			} 
			iList.push(exgene[i]);
			iList.push(nodeProb);
			iList.push(pathProb);
			geneList.push(iList);
		}
	}
	geneList.sort(function(a,b){
			if (a[2]<b[2]){
				return -1;
			}else if (a[2]>b[2]){
				return 1;
			}else{
				return 0;
			}
		});
	
	var keys=[];
	var goGeneCut=800;
	for(var i=0;(i<goGeneCut)&(i<geneList.length);i++){
		keys.push(geneList[i][0]);
	}
	panthergoInput(keys);
};

var panthergoInput=function(keys,species){
	var link_pre="http://pantherdb.org/webservices/go/overrep.jsp?input=";
	if (species==undefined){
		var link_suffix="&species=MOUSE";
	}else{
		var link_suffix="&species="+species;
	}
	
	keys=keys.join("\n");
	key=encodeURIComponent(keys);
	var link=link_pre+keys+link_suffix;
	link=encodeURI(link);
	var ww=open(link,'_blank','height=600,width=800,left=1200,top=200,scrollbars=yes');
}
//


var panthergoInput1=function(keys){
	var link_pre="http://pantherdb.org/geneListAnalysis.do?idField=";
	var link_suffix="&fileType=10&organism=Mus+musculus&dataset=Mus+musculus&resultType=2";
	keys=keys.join("+");
	key=encodeURIComponent(keys);
	var link=link_pre+keys+link_suffix;
	link=encodeURI(link);
	open(link,'_blank','height=600,width=800,left=1200,top=200,scrollbars=yes');
}


//toppgene go 
var toppgenego=function(node){
	var geneList=[];
	var geneIndex=node.genesInNode;
	for (var i in geneIndex){
		var iList=[];
		if (geneIndex[i]){
			var iex=genex[i][node.depth];
			var nodeProb=NormProb(iex,node.nodeMean,node.nodeSigma);
			var pathProb=0;
			var iNodes=getNodeList(exgene[i]);
			for (var iNode of iNodes){
				var iNodeEx=genex[i][iNode.__data__.depth];
				var iprob=NormProb(iNodeEx,iNode.__data__.nodeMean,iNode.__data__.nodeSigma)
				pathProb=pathProb+iprob;
			} 
			iList.push(exgene[i]);
			iList.push(nodeProb);
			iList.push(pathProb);
			geneList.push(iList);
		}
	}
	geneList.sort(function(a,b){
			if (a[2]<b[2]){
				return -1;
			}else if (a[2]>b[2]){
				return 1;
			}else{
				return 0;
			}
		});
	
	var keys=[]
	var goGeneCut=800;
	for(var i=0;(i<goGeneCut)&(i<geneList.length);i++){
		keys.push(geneList[i][0]);
	}
	toppgenegoInput(keys);

};

var toppgenegoInput=function(keys){
	var link_pre="https://toppgene.cchmc.org/CheckInput.action?query=TOPPFUN&type=HGNC_SYNONYMS&training_set="
	var link_suffix="";
	keys=keys.join("+");
	key=encodeURIComponent(keys);
	var link=link_pre+keys+link_suffix;
	link=encodeURI(link);
	open(link,'_blank','height=600,width=800,left=1200,top=200,scrollbars=yes');
}

//show methylation for all targets of top TFs associated to clicked node

var showTopTFMethyAll=function(node){
	var tfArray=node.ETF;
	var tfs=[];
	var topcut=20;
	topcut=document.getElementById("popupcut").innerHTML;
	topcut=parseInt(topcut);
	
	for (var i=0;i<topcut;i++){
		var tf=tfArray[i][0].split(" ")[0];
		tfs.push(tf);
	}
	var tftargets=[];
	for (var tf of tfs){
		targets=regTargetMap[tf];
		for (var target of targets){
			if (tftargets.indexOf(target)==-1){
				tftargets.push(target);
			}
		}
	}
	
	
	var tftargetsnode=[];
	for (var target of tftargets){
		if (exgene.indexOf(target)!=-1){
			tftargetsnode.push(target);
		}
	}
	
	showMethyGenes(tftargetsnode,"tf","(top20) node (All targets)"+node.nodeID);
}


//show methylation for targets (in node) of top 20 TFs associated to current node

var showTopTFMethy=function(node){
	var tfArray=node.ETF;
	var tfs=[];
	var topcut=20;
	topcut=document.getElementById("popupcut").innerHTML;
	topcut=parseInt(topcut);
	
	for (var i=0;i<topcut;i++){
		var tf=tfArray[i][0].split(" ")[0];
		tfs.push(tf);
	}
	var tftargets=[];
	for (var tf of tfs){
		targets=regTargetMap[tf];
		for (var target of targets){
			if (tftargets.indexOf(target)==-1){
				tftargets.push(target);
			}
		}
	}
	
	var nodeGenes=node.genes;
	var tftargetsnode=[];
	for (var target of tftargets){
		if (nodeGenes.indexOf(target)!=-1){
			tftargetsnode.push(target);
		}
	}
	
	showMethyGenes(tftargetsnode,"tf","(top20) node  "+node.nodeID);
}



//show methylation for given gene list
var showMethyGenes=function(nodeGenes,nameType,ID,geneplotdiv){
	var ScoreList=[];
	for (var gene of nodeGenes){
		var pks_gene=ScoreMethyGene(gene);
		if (pks_gene.length!=0){
			ScoreList.push(pks_gene.map(function(d){return d[1];}));
		}
	}
	
	var ScoreAvg=[];
	var methyTimes=[];
	for (var time of timePoints){
		if (time in methyscore){
			methyTimes.push(time);
		}
	}
	
	for (var timeIndex in methyTimes){
		var timeSum=0;
		if (methyTimes[timeIndex] in methyscore){
			for (var row of ScoreList){
				timeSum+=row[timeIndex];
			}
			var timeAvg=timeSum*1.0/ScoreList.length;
			ScoreAvg.push([methyTimes[timeIndex],timeAvg]);
		}
	}
	//
	if (nameType=="node"){
		var node=nodes[ID];
		var firstRow="Genes in node "+ID+";"+" path: "+node.path;
	}
	if (nameType=="tf"){
		var firstRow="target genes of Regulator "+ID;
	}
	
	
	var geneList=[[firstRow]];
	for (var gene of nodeGenes){
		geneList.push([gene]);
	}
	
	// plot
	if (geneplotdiv==undefined){
		var newW2 = open('','_blank','height=600,width=800,left=1000,top=100,scrollbars=yes');
		newW2.document.write("<head><link rel='stylesheet' type='text/css' href='style.css'></head><body></body>");
		var tfmethydiv=d3.select(newW2.document.body).append("div")
					.attr("id","tfmethy")
					.attr("width",800)
					.attr("height",700);
		tfmethydiv=createTable(tfmethydiv,"methytable",geneList,exploremethy);
		tfmethydiv.select('table')
		.selectAll("tr")
		.style('background-color',function(d){
				var target=d;
				for (var time of timePoints){
					if (time in methyscore){
						if (target in methyscore[time]){
							if (methyscore[time][target]>0.5){
								return 'gray';
							}
						}
					}
				}
			});
	}else{
		var tfmethydiv=geneplotdiv.append("div")
		.attr("id","tfmethy")
		.attr("width",800)
		.attr("height",700);
	}
				
	//plotHeatMap(ScoreList, 'methylation heatmap of '+firstRow);
	if (geneplotdiv==undefined){
		var newW2 = open('','_blank','height=800,width=1000,left=1000,top=200,scrollbars=yes');
		newW2.document.write("<head><link rel='stylesheet' type='text/css' href='style.css'></head><body></body>");
		geneplotdiv=d3.select(newW2.document.body).append("div");
	}
	gplotData(geneplotdiv,"",ScoreAvg,"Average Methylation for "+firstRow);
}



//show mouseover pop

var showmouseover=function(node){
	newWLC = open('','_blank','height=600,width=800,left=800,top=200,scrollbars=yes');
	newWLC.document.write("<body></body>");
	
	var showcut;
	showcut=document.getElementById("popupcut").innerHTML;
	showcut=parseInt(showcut);
	var etf=JSON.parse(JSON.stringify(node.ETF));
	var row=etf[0];
	var rowT1="TF	Num Total	Num Parent	Num Path	Expect Overall	Diff. Overall	Score Overall	Expect Split	Diff. Split	Score Split	% Split";
	var rowT2="TF	Num Total	Num Path	Expect Overall	Diff. Overall	Score Overall";
	var cetf=[];
	if (row.length==rowT1.split("\t").length){
		cetf.push(rowT1.split("\t"));
	}
	
	if (row.length==rowT2.split("\t").length){
		cetf.push(rowT2.split("\t"));
	}
	
	for (var i=0;i<showcut;i++){
		etf[i][0]=etf[i][0].split(" ")[0];
		cetf.push(etf[i]);
	}
	
	var genediv=d3.select(newWLC.document.body)
		.style("background","white")
		.append("div")
		.style("padding-left","50px")
		.style("padding-top","50px")
		.attr("width",400)
		.attr("height",600);
	createTFTable(genediv,"tftable",cetf,exploregene,node.depth);
	d3.event.preventDefault();
};


//show compare between different methods
var showCompare=function(etf,etf_no,etf_prote){
	newLCC = open('','_blank','height=600,width=600,left=1400,top=200,scrollbars=yes');
	newLCC.document.write("<body>TF comparison Table</body>");
	var showcut=20;
	if(etf_no[0]=='n'){
		etf_no=[];
	}
	if (etf_prote[0]=='n'){
		etf_prote=[];
	}
	var etf_list=[];
	for (var tf of etf){
		tf=tf.split(" ")[0];
		etf_list.push(tf);
	}
	etf_list=etf_list.slice(0,showcut);
	
	var etf_prote_list=[];
	for (var tf of etf_prote){
		tf=tf.split(" ")[0];
		etf_prote_list.push(tf);
	}
		
	var etf_no_list=[]
	for (var tf of etf_no){
		tf=tf.split(" ")[0];
		etf_no_list.push(tf);
	}
	
	var tdiv=d3.select(newLCC.document.body)
		.style("background","white")
		.append("div")
		.style("padding-left","50px")
		.style("padding-top","50px")
		.attr("width",400)
		.attr("height",600)
		.attr("class","div_table_tf_compare");
		
	etf=[]
	etf.push(['result_methylation_proteomics','result_proteomics','result_no_additional_data']);
	for (var i=0;i<etf_list.length;i++){
		var etf_i=[etf_list[i],etf_prote_list[i],etf_no_list[i]];
		etf.push(etf_i);
	}
	createCompareTable(tdiv,"tftable",etf);
};	

//single cell dropdown on change
var singlecelldropdownonchange=function(){
	var celltype=this.value;
	document.getElementById("singleCellType").value=celltype;
	exploresinglecell(celltype);
}

//sorted cell dropdown on change

var sortedcelldropdownonchange=function(){
	var celltype=this.value;
	document.getElementById("sortedCellType").value=celltype;
	exploresortedcell(celltype);
}


//tf dropdown menu on change
var tfdropdownonchange=function(){
	var tf=this.value;
	document.getElementById("tfName").value=tf;
	exploretf(tf);
}

//click dropdown menu on change

var clickdropdownonchange=function(){
	
};

//methylation dropdown menu on change
var methyTimedropdownonchange=function(){
	var time=this.value;
	var targets=Object.keys(methyscore[time]);
	createDropDown("Choose Gene","#methydropdownGenediv","methygenedropdown",targets,methyGenedropdownonchange);
	
}

var methyGenedropdownonchange=function(){
	var geneinput=this.value;
	document.getElementById("methyName").value=geneinput;
	exploremethy(geneinput);
}

/* search panel */

//explore path
/*
var explorepath=function(){
	
	// generate data X for visualization 
	var PathLabel=nodes[0].path.split(",");
	var allpath=[];
	for (var path of PathLabel){
		var PathList=[];
		for (var node of nodes){
			if (node.path.indexOf(path)!=-1){
				PathList.push(node)
			}
		}
		allpath.push(PathList);
	}
	
	var newW2 = open('','_blank','height=800,width=800,left=1000,top=200,scrollbars=yes');
	newW2.document.write("<head><link rel='stylesheet' type='text/css' href='style.css'></head><body></body>");
		
	var chart=d3.select(newW2.document.body).append("div")
					.append("svg")explorepath
					.attr("fill","black")
					.attr("width",800)
					.attr("height",800)
					//.style("border","1px solid")
					.style("margin-left","10px")
					.style("margin-right","10px")
					.style("padding-right","10px")
					.style("border","1px solid");
					
	var width = 600;
	var height = 600;
	var yoffset=100;
	
	var x = d3.scale.ordinal()
		.rangeRoundBands([0, width], .1);

	var y = d3.scale.linear()
		.range([height, 0]);
	
	var xAxis = d3.svg.axis()
	.scale(x)
	.orient("bottom");
	
	
	var yAxis = d3.svg.axis()
		.scale(y)
		.orient("left")
		.ticks(10);
	
	x.domain(nodes.map(function(d) { return d.nodetime; }));
	y.domain([-5,5]);
	
	var xrangeband=x.rangeBand();
	
	var lineFunction=d3.svg.line()
		.x(function(d) {return x(d.nodetime);})
		.y(function(d) {return y(d.nodeMean);})
		.interpolate("linear");
		
	
	var colorPane=["red","green","blue","purple","black"];
	
	var colorIndex=0;
	for (var path of allpath){
		chart.append("g")
		.attr("transform", function(d) { return "translate(" +100 + ","+(yoffset)+")"; })
		.append("path")
		.on("click",highlightPath)
		.attr("class","link")
		.attr("stroke",function(){
				return colorPane[colorIndex];
			})
		.attr("stroke-width",3)
		.attr('fill',"none")
		.attr("d",lineFunction(path));
		
		colorIndex+=1;
		colorIndex=colorIndex%colorPane.length;
	}
	
	colorIndex=0;
	for (var path of allpath){
		chart.append("g")
		.attr("transform", function(d) { return "translate(" +100 + ","+(yoffset)+")"; })
		.append("text")
		.attr("x",function(){
			 var leafnode=path[path.length-1];
			 var lx=x(leafnode.nodetime);
			 return lx+colorIndex*5;
			})
		.attr("y",function(){
			 var leafnode=path[path.length-1];
			 var ly=y(leafnode.nodeMean);
			 return ly;
			})
		.text(function(){
				var leafnode=path[path.length-1];
				return leafnode.path;
			})
		.style("fill",function(){
				return colorPane[colorIndex];
			})
		.style("font-size","12px");
		
		colorIndex+=1;
		colorIndex=colorIndex%colorPane.length;
	}
	
	chart.append("g")
	  .attr("class", "x axis")
	  .attr("transform", "translate(122," + (height+yoffset) + ")")
	  .call(xAxis)
	  .selectAll("text")
	  .attr("transform","rotate(90)translate(0,28)")
	  .style("stroke","black")
	  .attr("y",0)
	  .attr("x",0)
	  .attr("fill","black")
	  .style("text-anchor", "start")
	  .attr("dx","1em");
	
	
	
	chart.append("g")
	  .attr("class", "y axis")
	  .attr("transform", "translate(100," + (0+yoffset) + ")")
	  .call(yAxis)
	  .append("text")
	  .style("stroke","black")
	  .attr("dx","0em")
	  .attr("y",50)
	  .attr("transform","rotate(90)")
	  .attr("dy","0.71em")
	  .attr("fill","black")
	  .style("text-anchor","start")
	  .text("Path Expression Pattern");
	
}
* */

//explore path sankey diagram
var exploreopathfunction=function(){
	var checkstyle=d3.select('input[name="checkstyle"]:checked').property("value");
	if (checkstyle=="1"){
		exploreopathfunction1();
	}else{
		exploreopathfunction2();
	}
}


//explore path sankey diagram
//sankey figure style 1
var exploreopathfunction1=function(){
	var tfcut=document.getElementById("tfcutSelector").innerHTML;
	tfcut=parseInt(tfcut);
	
	var gocut=document.getElementById("goslider").innerHTML;
	var gocut=parseInt(gocut);
	
	var tfnodecut=10;
	var mirnodecut=5;
	
	var PathLabel=nodes[0].path.split(",");
	PathLabel.sort();
	var allpath=[];
	for (var path of PathLabel){
		var PathList=[];
		for (var node of nodes){
			if (node.path.indexOf(path)!=-1){
				PathList.push(node)
			}
		}
		allpath.push(PathList);
	}
	var leafs=allpath.map(function(d){return d[d.length-1];});
	var sankeyNodes=[]; //nodes
	var sankeyPaths=[]; //links
	// adding path
	for (var path of PathLabel){
		var inode={"name":path};
		sankeyNodes.push(inode);
	}
	
	// adding go terms
	var goTermList=[];
	var usedTerms=[];
	for (var leaf of leafs){
		var golist=leaf.goNode.slice(0,gocut);
		for (var goitem of golist){
			if (usedTerms.indexOf(goitem[1])==-1){
					goTermList.push(goitem);
					usedTerms.push(goitem[1]);
			}
		}
	}
	
	//adding go terms nodes
	goTermList=goTermList.map(function(d){return d[1];});
	//goTermList.sort();
	for (var goTerm of goTermList){
		var gonode={"name":goTerm};
		sankeyNodes.push(gonode);
	}
	
	// output: a dictionary of top TF/miRNAs for each path
	var AllPathTFList=[];
	var AllPathMIRList=[];
	var AllPathTFListD=[];
	var AllPathMIRListD=[];
	for (var path of allpath){
		var dpathtf={};
		var pathTFList=[];
		var dpathmir={};
		var pathMIRList=[];
		
		for (var node of path){
			var nodepath=node.path;
			if (nodepath.length==1 && node.ETF!=undefined){
				for (var tf of node.ETF.slice(0,tfcut)){
					var tfname=tf[0].split(" ")[0];
					if (tf.length<7){
						var pv=parseFloat(tf[5]);
					}else{
						var pv=parseFloat(tf[6]);
					}
					if (tfname.toUpperCase().indexOf("MIR-")==-1  && tfname.toUpperCase().indexOf("LET-")==-1){
						if (!(tfname in dpathtf)){
							dpathtf[tfname]=pv;
						}else{
							dpathtf[tfname]=Math.max(pv,dpathtf[tfname]);
						}
					}else{
						if (!(tfname in dpathmir)){
							dpathmir[tfname]=pv;
						}else{
							dpathmir[tfname]=Math.max(pv,dpathmir[tfname]);
						}
					}
					
				}
				
			}	
		}
		//
		for (var tf in dpathtf){
			pathTFList.push([tf,dpathtf[tf]]);
		}
		for (var mir in dpathmir){
			pathMIRList.push([mir,dpathmir[mir]]);
		}
		
		pathTFList.sort(function(a,b){return a[1]-b[1];});
		pathMIRList.sort(function(a,b){return a[1]-b[1];});
		
		pathTFList=pathTFList.slice(0,tfnodecut);
		pathMIRList=pathMIRList.slice(0,mirnodecut);
		
		AllPathTFListD.push(pathTFList);
		AllPathMIRListD.push(pathMIRList);
		//
		pathTFList=pathTFList.map(function(d){return d[0];});
		pathMIRList=pathMIRList.map(function(d){return d[0];});
		AllPathTFList.push(pathTFList);
		AllPathMIRList.push(pathMIRList);
		
		
	}
	
	var gonumber=sankeyNodes.length;
	//adding TF nodes 
	for (var i=0;i<leafs.length;i++){
		var TFList=AllPathTFList[i];
		var tfnode={"name": TFList};
		sankeyNodes.push(tfnode);
	}
	
	var mirnodenumber=sankeyNodes.length;
	//adding mir nodes
	for (var i=0;i<leafs.length;i++){
		var MIRList=AllPathMIRList[i];
		var mirnode={"name": MIRList};
		sankeyNodes.push(mirnode);
	}
	
	//adding sankey links
	var sankeyNodeNames=sankeyNodes.map(function(d){return d.name;});
	var sankeyPathDetails=[];
	
	// adding go links
	for (var i=0;i<leafs.length;i++){
		var igo=leafs[i].goNode.slice(0,gocut);
		var source=sankeyNodeNames.indexOf(leafs[i].path);
		for (var j=0;j<igo.length;j++){
			var target=sankeyNodeNames.indexOf(igo[j][1]);
			var value=1.0/(j+1);
			sankeyPaths.push({"source":source,"target":target,"value":value});
			sankeyPathDetails.push([igo[j]]);
		}
	}
	
	//adding tf links
	for (var i=0;i<leafs.length;i++){
		var pathLabel=leafs[i].path;
		var pathTFList=AllPathTFList[i];
		var ipathtfs=AllPathTFListD[i];
	
		var source=sankeyNodeNames.indexOf(pathTFList);
		var target=sankeyNodeNames.indexOf(pathLabel);
		if (pathTFList.length>0){
			sankeyPaths.push({"source":source, "target":target, "value":1});
			sankeyPathDetails.push(ipathtfs);
		}
	}
	
	// adding mir links
	for (var i=0;i<leafs.length;i++){
		var pathLabel=leafs[i].path;
		var pathMIRList=AllPathMIRList[i];
		var ipathmirs=AllPathMIRListD[i];
		var source=sankeyNodeNames.indexOf(pathMIRList);
		var target=sankeyNodeNames.indexOf(pathLabel);
		if (pathMIRList.length>0){
			sankeyPaths.push({"source":source, "target":target, "value":1});
			sankeyPathDetails.push(ipathmirs);
		}
	}
	
	//
	//
	var sankeyW=1000
	var sankeyH=Math.max(gocut,tfnodecut,mirnodecut)*200
		
	var sankeyData={"nodes":sankeyNodes,"links":sankeyPaths};
	var newW2 = open('','_blank','height=1400,width=1800,left=200,top=200,scrollbars=yes');
	newW2.document.write("<head><link rel='stylesheet' type='text/css' href='style.css'></head><body></body>");
	var sankeyplotdiv=d3.select(newW2.document.body).append("div");
	var pcolor=d3.scale.category20();
	var sankeychart=sankeyplotdiv.append("svg")
		.attr("id","sankeychart")
		.attr("width",sankeyW)
		.attr("height",sankeyH)
		.chart("Sankey"),
		color=d3.scale.category20();
	
				
	sankeychart
		.nodeWidth(10)
		.nodePadding(10)
		.alignLabel(function(n){
			if (sankeyNodeNames.indexOf(n.name)<gonumber){
				return 'start';
			}else{
				return 'end';
			}
		})
		.colorNodes(function(name,node){
			var cindex=sankeyNodeNames.indexOf(name)
			if (cindex<gonumber){
				return pcolor(cindex);
			}else if (cindex<mirnodenumber){
				return pcolor(cindex-gonumber);
			}else{
				return pcolor(cindex-mirnodenumber);
			}
		})
		.iterations(0)
		.spread(true)
		.draw(sankeyData);
	
	// move the plot to the right (leave space for TF/miRNAs)
	var rightmove=600	
	d3.select(newW2.document.body).select("#sankeychart")
		.selectAll("rect")
		.attr("transform","translate("+rightmove+")");
	
	d3.select(newW2.document.body).select("#sankeychart")
		.selectAll("text")
		.attr("transform","translate("+rightmove+")");
	
	d3.select(newW2.document.body).select("#sankeychart")
		.selectAll(".link")
		.attr("transform","translate("+rightmove+")");
	
	//extend the sankey svg canvas
	d3.select(newW2.document.body).select("#sankeychart")
	.style("width", sankeyW+1000)
	.style("height",sankeyH+200);
		
	
	//
	var rects=d3.select(newW2.document.body).select("#sankeychart")
		.selectAll("rect")[0];
	
	var links=d3.select(newW2.document.body).select("#sankeychart")
		.selectAll(".link")[0];
		
	
	sankeychart.on('node:mouseover',function(node){
		for (var li of links){
			if (li.__data__.source==node){
				li.style.strokeOpacity="0.5";
			}
		}
	});
	
	sankeychart.on('node:click',function(node){
		//var nodeFunctions=[["GO Term ID","GO Term Name", "p-value"]];
		var nodeFunctions=[];
		for (var path_data of node.sourceLinks){
			var path_index=sankeyPaths.indexOf(path_data);
			var path_detail=sankeyPathDetails[path_index];
			nodeFunctions.push(path_detail);
		}
		var newW3 = open('','_blank','height=600,width=800,left=200,top=200,scrollbars=yes');
			newW3.document.write("<head><link rel='stylesheet' type='text/css' href='style.css'></head><body></body>");
			var genediv=d3.select(newW3.document.body)
			.style("background","white")
			.append("div")
			.style("padding-left","50px")
			.style("padding-top","50px")
			.attr("width",400)
			.attr("height",600);
		createTable(genediv,"sankeyNodetable",nodeFunctions,null);
	});
	
	sankeychart.on('node:mouseout',function(node){
		for (var li of links){
			if (li.__data__.source==node){
				li.style.strokeOpacity="0.2";
			}
		}
	});
			
	//	
	d3.select(newW2.document.body).select("#sankeychart")
		.selectAll(".link")
		.on("mouseover",function(d,i){
			this.style.strokeOpacity="0.5";
			for (var ri of rects){
				if (ri.__data__==d.target){
					ri.style.fillOpacity="0";
				}
				if (ri.__data__==d.source){
					ri.style.fillOpacity="0";
				}
			}
			
		})
		.on("mouseout",function(d,i){
			this.style.strokeOpacity="0.2";
			for (var ri of rects){
				if (ri.__data__==d.target){
					ri.style.fillOpacity="0.9";
				}
				if (ri.__data__==d.source){
					ri.style.fillOpacity="0.9";
				}
			}
			
		})
		.on("click",function(d,i){
			var path_data=d;
			var path_index=sankeyPaths.indexOf(path_data);
			var path_detail=sankeyPathDetails[path_index];
			var newW3 = open('','_blank','height=600,width=800,left=200,top=200,scrollbars=yes');
			newW3.document.write("<head><link rel='stylesheet' type='text/css' href='style.css'></head><body></body>");
			var genediv=d3.select(newW3.document.body)
			.style("background","white")
			.append("div")
			.style("padding-left","50px")
			.style("padding-top","50px")
			.attr("width",400)
			.attr("height",600);
			var path_detailList=[];
			for (var item of path_detail){
				path_detailList.push(item);
			}
			createTable(genediv,"sankeytable",path_detailList,exploretf);	
			})
			
		.append("svg:title")
          .text(function(d, i) { 
				var path_data=d;
				var path_index=sankeyPaths.indexOf(path_data);
				var path_detail=sankeyPathDetails[path_index];
				return path_detail;
			   });
	
}

//explore path function (sankey figure style 2)
var exploreopathfunction2=function(){
	var tfcut=document.getElementById("tfcutSelector").innerHTML;
	tfcut=parseInt(tfcut);
	
	var gocut=document.getElementById("goslider").innerHTML;
	var gocut=parseInt(gocut);
	
	var tfnodecut=20;
	var mirnodecut=10;
	
	var PathLabel=nodes[0].path.split(",");
	PathLabel.sort();
	var allpath=[];
	for (var path of PathLabel){
		var PathList=[];
		for (var node of nodes){
			if (node.path.indexOf(path)!=-1){
				PathList.push(node)
			}
		}
		allpath.push(PathList);
	}
	var leafs=allpath.map(function(d){return d[d.length-1];});
	var sankeyNodes=[]; //nodes
	var sankeyPaths=[]; //links
	// adding path
	for (var path of PathLabel){
		var inode={"name":path};
		sankeyNodes.push(inode);
	}
	
	// adding go terms
	var goTermList=[];
	var usedTerms=[];
	for (var leaf of leafs){
		var golist=leaf.goNode.slice(0,gocut);
		for (var goitem of golist){
			if (usedTerms.indexOf(goitem[1])==-1){
					goTermList.push(goitem);
					usedTerms.push(goitem[1]);
			}
		}
	}
		
	//adding go terms nodes
	goTermList=goTermList.map(function(d){return d[1];});
	//goTermList.sort();
	for (var goTerm of goTermList){
		var gonode={"name":goTerm};
		sankeyNodes.push(gonode);
	}
	var gonumber=sankeyNodes.length;
	
	// output: a dictionary of top TF/miRNAs for each path
	var AllPathTFList=[];
	var AllPathMIRList=[];
	var AllPathTFListD=[];
	var AllPathMIRListD=[];
	
	for (var path of allpath){
		var dpathtf={};
		var pathTFList=[];
		var dpathmir={};
		var pathMIRList=[];
		
		for (var node of path){
			var nodepath=node.path;
			if (nodepath.length==1 && node.ETF!=undefined){
				for (var tf of node.ETF.slice(0,tfcut)){
					var tfname=tf[0].split(" ")[0];
					if (tf.length<7){
						var pv=parseFloat(tf[5]);
					}else{
						var pv=parseFloat(tf[6]);
					}
					if (tfname.toUpperCase().indexOf("MIR-")==-1  && tfname.toUpperCase().indexOf("LET-")==-1){
						if (!(tfname in dpathtf)){
							dpathtf[tfname]=pv;
						}else{
							dpathtf[tfname]=Math.max(pv,dpathtf[tfname]);
						}
					}else{
						if (!(tfname in dpathmir)){
							dpathmir[tfname]=pv;
						}else{
							dpathmir[tfname]=Math.max(pv,dpathmir[tfname]);
						}
					}
					
				}
				
			}	
		}
		//
		for (var tf in dpathtf){
			pathTFList.push([tf,dpathtf[tf]]);
		}
		for (var mir in dpathmir){
			pathMIRList.push([mir,dpathmir[mir]]);
		}
		
		pathTFList.sort(function(a,b){return a[1]-b[1];});
		pathMIRList.sort(function(a,b){return a[1]-b[1];});
		
		AllPathTFListD.push(pathTFList);
		AllPathMIRListD.push(pathMIRList);
		
		pathTFList=pathTFList.map(function(d){return d[0];});
		pathMIRList=pathMIRList.map(function(d){return d[0];});
		
		AllPathTFList.push(pathTFList);
		AllPathMIRList.push(pathMIRList);
	}
	
	// adding TF nodes
	

	usedTFs=[];
	for (var tTFList of AllPathTFList){
		for (var tf of tTFList.slice(0,tfnodecut)){
			var tfindex=usedTFs.indexOf(tf);
			if (tfindex==-1){
				var tfnode={"name":tf};
				sankeyNodes.push(tfnode);
				usedTFs.push(tf);
			}
		}
	}
	
	// adding miRNA nodes
	usedMIRs=[];
	
	for (var tMIRList of AllPathMIRList){
		for (var mir of tMIRList.slice(0,tfnodecut)){
			if (usedMIRs.indexOf(mir)==-1){
				var mirnode={"name":mir};
				sankeyNodes.push(mirnode);
				usedMIRs.push(mir);
			}
		}
	}
	
	//adding sankey links
	var sankeyNodeNames=sankeyNodes.map(function(d){return d.name;});
	var sankeyPathDetails=[];
	
	// adding go links
	for (var i=0;i<leafs.length;i++){
		var igo=leafs[i].goNode.slice(0,gocut);
		var source=sankeyNodeNames.indexOf(leafs[i].path);
		for (var j=0;j<igo.length;j++){
			var target=sankeyNodeNames.indexOf(igo[j][1]);
			var value=1.0/(j+1);
			sankeyPaths.push({"source":source,"target":target,"value":value});
			sankeyPathDetails.push(igo[j]);
		}
	}
	
	// adding tf links
	for (var i=0;i<leafs.length;i++){
		var pathLabel=leafs[i].path;
		for (var tf of usedTFs){
			var source=sankeyNodeNames.indexOf(tf);
			var tPathList=AllPathTFList[i];
			var tpathListD=AllPathTFListD[i];
			var tpathIndex=tPathList.indexOf(tf);
			if (tpathIndex!=-1){
				var target=sankeyNodeNames.indexOf(pathLabel);
				var tfd=tpathListD[tpathIndex];
				sankeyPaths.push({"source":source, "target":target, "value":1});
				sankeyPathDetails.push(tfd);
			}	
		}
	}
	
	// adding mir links
	for (var i=0;i<leafs.length;i++){
		var pathLabel=leafs[i].path;
		for (var mir of usedMIRs){
			var source=sankeyNodeNames.indexOf(mir);
			var tPathList=AllPathMIRList[i];
			var tpathListD=AllPathMIRListD[i];
			var tpathListIndex=tPathList.indexOf(mir);
			if (tpathListIndex!=-1){
				var target=sankeyNodeNames.indexOf(pathLabel);
				var mird=tpathListD[tpathListIndex];
				sankeyPaths.push({"source":source, "target":target, "value":1});
				sankeyPathDetails.push(mird);
			}	
		}
	}
	
	//
	var sankeyW=1000
	var sankeyH=Math.max(gocut,tfnodecut,mirnodecut)*200
	
	var sankeyData={"nodes":sankeyNodes,"links":sankeyPaths};
	var newW2 = open('','_blank','height=1400,width=1200,left=1000,top=200,scrollbars=yes');
	newW2.document.write("<head><link rel='stylesheet' type='text/css' href='style.css'></head><body></body>");
	var sankeyplotdiv=d3.select(newW2.document.body).append("div");
	var sankeychart=sankeyplotdiv.append("svg")
		.attr("id","sankeychart")
		.attr("width",sankeyW)
		.attr("height",sankeyH)
		.chart("Sankey"),
		color=d3.scale.category20();
				
	sankeychart
		.nodeWidth(10)
		.nodePadding(10)
		.spread(true)
		.alignLabel(function(n){
			if (sankeyNodeNames.indexOf(n.name)<gonumber){
				return 'start';
			}else{
				return 'end';
			}
		})
		.iterations(0)
		.draw(sankeyData);
		
	
	
	// move the plot to the right (leave space for TF/miRNAs
	var rightmove=200	
	d3.select(newW2.document.body).select("#sankeychart")
		.selectAll("rect")
		.attr("transform","translate("+rightmove+")");
	
	d3.select(newW2.document.body).select("#sankeychart")
		.selectAll("text")
		.attr("transform","translate("+rightmove+")");
	
	d3.select(newW2.document.body).select("#sankeychart")
		.selectAll(".link")
		.attr("transform","translate("+rightmove+")");
	
	var rects=d3.select(newW2.document.body).select("#sankeychart")
		.selectAll("rect")[0];
	
	var links=d3.select(newW2.document.body).select("#sankeychart")
		.selectAll(".link")[0];
		
	//
	d3.select(newW2.document.body).select("#sankeychart")
	.style("width", sankeyW+1000)
	.style("height",sankeyH+200);
	

	sankeychart.on('node:mouseover',function(node){
		for (var li of links){
			if (li.__data__.source==node){
				li.style.strokeOpacity="0.5";
			}
		}
	});
	
	sankeychart.on('node:click',function(node){
		var nodeFunctions=[["GO Term ID","GO Term Name", "p-value"]];
		for (var path_data of node.sourceLinks){
			var path_index=sankeyPaths.indexOf(path_data);
			var path_detail=sankeyPathDetails[path_index];
			nodeFunctions.push(path_detail);
		}
		var newW3 = open('','_blank','height=600,width=800,left=200,top=200,scrollbars=yes');
			newW3.document.write("<head><link rel='stylesheet' type='text/css' href='style.css'></head><body></body>");
			var genediv=d3.select(newW3.document.body)
			.style("background","white")
			.append("div")
			.style("padding-left","50px")
			.style("padding-top","50px")
			.attr("width",400)
			.attr("height",600);
		createTable(genediv,"sankeyNodetable",nodeFunctions,null);
	});
	
	sankeychart.on('node:mouseout',function(node){
		for (var li of links){
			if (li.__data__.source==node){
				li.style.strokeOpacity="0.2";
			}
		}
	});
		
	d3.select(newW2.document.body).select("#sankeychart")
		.selectAll(".link")
		.on("mouseover",function(d,i){
			this.style.strokeOpacity="0.5";
			for (var ri of rects){
				if (ri.__data__==d.target){
					ri.style.fillOpacity="0";
				}
				if (ri.__data__==d.source){
					ri.style.fillOpacity="0";
				}
			}
			
		})
		.on("mouseout",function(d,i){
			this.style.strokeOpacity="0.2";
			for (var ri of rects){
				if (ri.__data__==d.target){
					ri.style.fillOpacity="0.9";
				}
				if (ri.__data__==d.source){
					ri.style.fillOpacity="0.9";
				}
			}
			
		})
		.on("click",function(d,i){
			var path_data=d;
			var path_index=sankeyPaths.indexOf(path_data);
			var path_detail=sankeyPathDetails[path_index];
			var newW3 = open('','_blank','height=600,width=800,left=200,top=200,scrollbars=yes');
			newW3.document.write("<head><link rel='stylesheet' type='text/css' href='style.css'></head><body></body>");
			var genediv=d3.select(newW3.document.body)
			.style("background","white")
			.append("div")
			.style("padding-left","50px")
			.style("padding-top","50px")
			.attr("width",400)
			.attr("height",600);
			var path_detailList=[];
			for (var item of path_detail){
				path_detailList.push([item]);
			}
			createTable(genediv,"sankeytable",path_detailList,exploretf);	
			})
			
		.append("svg:title")
          .text(function(d, i) { 
				var path_data=d;
				var path_index=sankeyPaths.indexOf(path_data);
				var path_detail=sankeyPathDetails[path_index];
				return path_detail;
			   });
	
}


//explore path

var explorepath=function(){
	
	// generate data X for visualization 
	var PathLabel=nodes[0].path.split(",");
	var allpath=[];
	for (var path of PathLabel){
		var PathList=[];
		for (var node of nodes){
			if (node.path.indexOf(path)!=-1){
				PathList.push(node)
			}
		}
		allpath.push(PathList);
	}
	var pathX=[]
	var LT=allpath[0].length;
	for (var t=0;t<LT;t++){
		var ext=[timePoints[t]];
		for (var path of allpath){
			ext.push(path[t].nodeMean);
		}
		pathX.push(ext);
	}
	var xcols=[];
	
	/*
	for (var di=0;di<allpath.length;di++){
		var chr=String.fromCharCode(65+di)
		xcols.push(chr);
	}
	* */
	var newW2 = open('','_blank','height=800,width=1000,left=1000,top=200,scrollbars=yes');
	newW2.document.write("<head><link rel='stylesheet' type='text/css' href='style.css'></head><body></body>");
	var geneplotdiv=d3.select(newW2.document.body).append("div");
	gplotData(geneplotdiv,'path',pathX,'path expression pattern',PathLabel);
}


//highlightPath on click

var highlightPath=function(){
	d3.select(this.ownerSVGElement).selectAll('path')
	.attr("stroke-width",3);
	var thisIndex=d3.select(this.ownerSVGElement).selectAll('path')[0].indexOf(this);
	
	d3.select(this)
	.attr("stroke-width",5);
	
	d3.select(this.ownerSVGElement).selectAll('text')
	.style("font-size",function(d,i){
		if (i==thisIndex){
			return '20px';
		}return '12px';
	});
}
//view methytrack in UCSC genome broswer

var viewmethytrack=function(){
	var pre_link="http://genome.ucsc.edu/cgi-bin/hgTracks?";
	var db="mm10";
	var dbs=document.getElementById("datareferenceselect").value;
	if (dbs!=undefined){
		db=dbs;
	}
	

	var type=document.getElementById("datatypeselect").value;
	var bigdata=document.getElementById("viewmethy").value;
	var name=bigdata.split("/");
	var name=name[name.length-1];
	var link=pre_link+"db="+db+"&hgct_customText=track%20type="+type+"%20name="+name+"%20visibility=full%20bigDataUrl="+bigdata;
	open(link,'_blank','height=600,width=800,left=1200,top=200,scrollbars=yes');
	
}

//wild explore methylation for targets for given TF
var wildexploretfmethy=function(tfinput){
	if (tfinput===undefined){
		var tfinput=document.getElementById("methyNameTF").value;
	}
	tfinput=tfinput.toUpperCase();
	
	var GeneList=[];
	var localReg=reg.map(function(d){return d.toUpperCase();});
	for(var gi of localReg){
		if (gi.indexOf(tfinput)!=-1){
			GeneList.push([gi]);
		}
	}
	
	if (GeneList.length==1 && GeneList[0]==tfinput){
		exploretfmethy(tfinput);
	} else{
		newWWG = open('','_blank','height=600,width=800,left=1400,top=200,scrollbars=yes');
		newWWG.document.write("<body>matches </body>");
		
		var genediv=d3.select(newWWG.document.body)
			.style("background","white")
			.append("div")
			.style("padding-left","50px")
			.style("padding-top","50px")
			.attr("width",400)
			.attr("height",600)
			.attr("class","div_matched_mirs");
		createTable(genediv,"mtable",GeneList,exploretfmethy);
	}
}



//explore methylation of targets for given TF
var exploretfmethy=function(tfinput,geneplotdiv){
	if (tfinput==undefined){
		tfinput=document.getElementById("methyNameTF").value;
	}
	
	tfnode=document.getElementById("regulatormethynodedropdown").value;
	var tftargets=regTargetMap[tfinput];
	if (tftargets===undefined){
		tftargets=regTargetMap[tfinput.toUpperCase()];
	}
	var targetList=[];
	
	if (tfnode=='All' || tfnode=="0"){
		for (var target of tftargets){
			targetList.push(target);
		}
	}else{
		var chosenNode=nodes[parseInt(tfnode)];
		for (var target of tftargets){
			if(chosenNode.genes.indexOf(target)!=-1){
				targetList.push(target);
			}
		}
	}
	showMethyGenes(targetList,'tf',tfinput+" in "+tfnode+ " node(s)",geneplotdiv);
}

//wild explore gene methylation

var wildexploremethy=function(geneinput){
	
	if (geneinput===undefined){
		geneinput=document.getElementById("methyName").value;
	}
	
	geneinput=geneinput.toUpperCase();
	var GeneList=[];
	var methyscoreKeys=Object.keys(Object.values(methyscore)[1]);
	for(var gi of methyscoreKeys){
		if (gi.indexOf(geneinput)!=-1){
			GeneList.push([gi]);
		}
	}
	if (GeneList.length==1 && GeneList[0]==geneinput){
		exploremethy(geneinput);
	}else{
		newWWG = open('','_blank','height=600,width=800,left=1400,top=200,scrollbars=yes');
		newWWG.document.write("<body>matches </body>");
		
		var genediv=d3.select(newWWG.document.body)
			.style("background","white")
			.append("div")
			.style("padding-left","50px")
			.style("padding-top","50px")
			.attr("width",400)
			.attr("height",600)
			.attr("class","div_matched_genes");
		createTable(genediv,"mgenetable",GeneList,exploremethy);
	}
}

//explore gene methylation
var exploremethy=function(geneinput,geneplotdiv){
	if (geneinput===undefined){
		geneinput=document.getElementById("methyName").value;
	}
	var ScoreList=ScoreMethyGene(geneinput);
	if (geneplotdiv==undefined){
		var newW2 = open('','_blank','height=800,width=1000,left=1000,top=200,scrollbars=yes');
		newW2.document.write("<head><link rel='stylesheet' type='text/css' href='style.css'></head><body></body>");
		geneplotdiv=d3.select(newW2.document.body).append("div");
	}
	if (ScoreList.length!=0){
		gplotData(geneplotdiv,geneinput,ScoreList,"Methylation Score of "+geneinput);
	}
}

//explore diff methylation (explore methylation difference)

var explorediffmethy=function(){
	var methytime1=d3.select("#diffmethytimedropdown1").property("value");
	var methytime2=d3.select("#diffmethytimedropdown2").property("value");
	var methyGenes1=methyscore[methytime1];
	var methyGenes2=methyscore[methytime2];
	
	var methyGeneList=[];
	var allGeneNames=geneabex.map(function(d){
			return d[0];
		});

	for (var gene of exgene){
		var gscore1=methyGenes1[gene.toUpperCase()];
		var gscore2=methyGenes2[gene.toUpperCase()];
		if (gscore1!=undefined && gscore2!=undefined){
			methyGeneList.push([gene,gscore1,gscore2,(gscore2-gscore1).toFixed(2)]);
		}
	}
	
	var mcut=0.25
	methyGeneList=methyGeneList.sort(function(a,b){return b[3]-a[3];});
	methyGeneListUp=methyGeneList.filter(function(d){
			if (d[3]>=mcut){
				return true;
			}else{
				return false;
			}
		});
	methyGeneListUp=[["gene","methylation score time 1","methylation score time 2","methylation score difference"]].concat(methyGeneListUp);
	
	methyGeneListDown=methyGeneList.filter(function(d){
			if (d[3]<=-1*mcut){
				return true;
			}else{
				return false;
			}
		});
	
	methyGeneListDown=methyGeneListDown.sort(function(a,b){return a[3]-b[3];});
	methyGeneListDown=[["gene","methylation score time 1","methylation score time 2","methylation score difference"]].concat(methyGeneListDown);
	var newW_methydiff = open('','_blank','height=600,width=600,left=1400,top=200,scrollbars=yes');
	newW_methydiff.document.write("<body></body>");
	var genediv=d3.select(newW_methydiff.document.body)
		.style("background","white")
		.append("div")
		.style("padding-left","50px")
		.style("padding-top","50px")
		.attr("width",400)
		.attr("height",600)
		.attr("class","diffmethy_table_edge");
	createTable(genediv,"methydifftableup",methyGeneListUp,exploremethy);
	
	var genediv2=d3.select(newW_methydiff.document.body)
	.style("background","white")
	.append("div")
	.style("padding-left","50px")
	.style("padding-top","50px")
	.attr("width",400)
	.attr("height",600)
	.attr("class","diffmethy_table_edge");
	createTable(genediv2,"methydifftabledown",methyGeneListDown,exploremethy);
	
	//explore go
	
	var shiftclickdropdownValue=document.getElementById("shiftclickdropdown").value;
	var topgenes=methyGeneListUp.map(function(d){return d[0];});
	var downgenes=methyGeneListDown.map(function(d){return d[0];});
	
	if (shiftclickdropdownValue==="ToppGene"){
		toppgenegoInput(topgenes);
		toppgenegoInput(downgenes);
	}else{
		panthergoInput(topgenes);
		panthergoInput(downgenes);
	}
}

//explore single cell
var exploresinglecell=function(celltype){
	resetNode();
	if (celltype===undefined){
		var celltype=d3.select("#singleCellType").property("value").toUpperCase();
	}
	celltype=celltype.toUpperCase();
	
	var N=exgene.length;
	var AllNodes=d3.select("svg").selectAll("g.node")[0];
	var scResultList=[];
	var allTimes=[];
	for (var node of AllNodes){
		if (allTimes.indexOf(node.__data__.nodetime)==-1){
			allTimes.push(node.__data__.nodetime);
		}
	}
	
	for (var cell of singleCells){
		var cell_time=cell[0];
		var cell_type=cell[1].toUpperCase();
		var cell_target=cell[2].split(",").map(function(d){return d.toUpperCase();});
		cell_target=cell_target.filter(function(d){
				if (exgene.indexOf(d)!=-1){
					return true;
				}else{
					return false;
				}
			});
		if (cell_type==celltype){
			var p=cell_target.length*1.0/N;
			var pv_cut=0.05;
			var timeNodes=[];
			if (allTimes.indexOf(cell_time)!=-1){
				for (var node of AllNodes){
					if (node.__data__.nodetime==cell_time){
						timeNodes.push(node);
					}
				}
			} else{
				timeNodes=AllNodes;
			}
			
			for (var node of timeNodes){
				var ov_tar=cell_target.filter(function(d){
					if (node.__data__.genes.indexOf(d)!=-1){
						return true;
					}else{
						return false;
					}
				});	
				var pv=1-pbinom(ov_tar.length-1,node.__data__.genes.length,p);
				if (pv<pv_cut){
					scResultList.push([node,pv,ov_tar]);
				}
			}
		}
	}
	
	var color=d3.scale.linear();
	color.domain([1,2000]);
	color.range(["white","red"]);
	
	var colorLowerBound=600;
	var chosenNodes=scResultList.map(function(d){return d[0];});
	
	for (var ListElement of scResultList){
		var colorValue=ListElement[1];
		if (pv_cut/colorValue>2000){
			colorValue=2000;
		}else{
			colorValue=pv_cut/colorValue;
			colorValue=Math.max(colorLowerBound,colorValue);
		}
		d3.select(ListElement[0]).selectAll("circle")
		.attr("fill",color(colorValue));
	}
	return scResultList;
}


//explore gene enrichment

var explorenrichment=function(genelist){
	resetNode();
	if (genelist===undefined){
		var genelist=d3.select("#genelistarea").property("value");
	}
	
	genelist=genelist.split(/[\r?\n]|\,|\ |\t+/g);
	console.log(genelist);
	genelist=genelist.map(function(d){return d.toUpperCase();});
	genelist=genelist.filter(function(d){
		if (exgene.indexOf(d)!=-1){
			return true;
		}else{
			return false;
		}
	});
	var p=genelist.length*1.0/exgene.length;
	var pv_cut=0.05;
	var AllNodes=d3.select("svg").selectAll("g.node")[0];
	var scResultList=[];

	for (var node of AllNodes){
		var ov=genelist.filter(function(d){
					if (node.__data__.genes.indexOf(d)!=-1){
						return true;
					}else{
						return false;
					}
				});
					
		var pv=1-pbinom(ov.length-1,node.__data__.genes.length,p);
		if (pv<pv_cut){
			scResultList.push([node,pv,ov]);
		}
	}
	var color=d3.scale.linear();
		color.domain([1,2000]);
		color.range(["white","red"]);
	
	
	var colorLowerBound=600;
	var chosenNodes=scResultList.map(function(d){return d[0];});
	for (var ListElement of scResultList){
		var colorValue=ListElement[1];
		if (pv_cut/colorValue>2000){
			colorValue=2000;
		}else{
			colorValue=pv_cut/colorValue;
			colorValue=Math.max(colorLowerBound,colorValue);
		}
		d3.select(ListElement[0]).selectAll("circle")
		.attr("fill",color(colorValue));
	}
	
	var ResultList=[["nodeID","# of overlapping genes", "p-value", "overlapping genes"]];
	for (var res of scResultList){
		ResultList.push([res[0].__data__.nodeID,res[2].length,res[1],res[2]]);
	}
	
	newWWG = open('','_blank','height=600,width=800,left=1400,top=200,scrollbars=yes');
	newWWG.document.write("<body>matches </body>");
	
	var ovdiv=d3.select(newWWG.document.body)
		.style("background","white")
		.append("div")
		.style("padding-left","50px")
		.style("padding-top","50px")
		.attr("width",400)
		.attr("height",600);
	createTable(ovdiv,"ovtable",ResultList,exploregene);
	return scResultList;
		
}



//explore sorted cell
var exploresortedcell=function(celltype){
	resetNode();
	if (celltype===undefined){
		var celltype=d3.select("#sortedCellType").property("value").toUpperCase();
	}
	celltype=celltype.toUpperCase();
	
	var N=exgene.length;
	var AllNodes=d3.select("svg").selectAll("g.node")[0];
	var scResultList=[];
	var allTimes=[];
	for (var node of AllNodes){
		if (allTimes.indexOf(node.__data__.nodetime)==-1){
			allTimes.push(node.__data__.nodetime);
		}
	}
	
	for (var cell of sortedCells){
		var cell_time=cell[0];
		var cell_type=cell[1].toUpperCase();
		var cell_target=cell[2].split(",").map(function(d){return d.toUpperCase();});
		cell_target=cell_target.filter(function(d){
				if (exgene.indexOf(d)!=-1){
					return true;
				}else{
					return false;
				}
			});
		if (cell_type==celltype){
			var p=cell_target.length*1.0/N;
			var pv_cut=0.05;
			var timeNodes=[];
			if (allTimes.indexOf(cell_time)!=-1){
				for (var node of AllNodes){
					if (node.__data__.nodetime==cell_time){
						timeNodes.push(node);
					}
				}
			} else{
				timeNodes=AllNodes;
			}
			
			for (var node of timeNodes){
				var ov_tar=cell_target.filter(function(d){
					if (node.__data__.genes.indexOf(d)!=-1){
						return true;
					}else{
						return false;
					}
				});	
				var pv=1-pbinom(ov_tar.length-1,node.__data__.genes.length,p);
				if (pv<pv_cut){
					scResultList.push([node,pv,ov_tar]);
				}
			}
		}
	}
	
	var color=d3.scale.linear();
		color.domain([1,2000]);
		color.range(["white","red"]);
	
	var colorLowerBound=600;
	var chosenNodes=scResultList.map(function(d){return d[0];});
	
	for (var ListElement of scResultList){
		var colorValue=ListElement[1];
		if (pv_cut/colorValue>2000){
			colorValue=2000;
		}else{
			colorValue=pv_cut/colorValue;
			colorValue=Math.max(colorLowerBound,colorValue);
		}
		d3.select(ListElement[0]).selectAll("circle")
		.attr("fill",color(colorValue));
	}
	return scResultList;
}

//wild explore regulator targets
var wildexploretftargets=function(tfinput){
	if (tfinput==undefined){
		var tfinput=d3.select("#tfTarget").property("value");
	}
	tfinput=tfinput.toUpperCase();
	var GeneList=[];
	var localReg=reg.map(function(d){return d.toUpperCase();});
	for(var gi of localReg){
		if (gi.indexOf(tfinput)!=-1){
			GeneList.push([gi]);
		}
	}
	
	if (GeneList.length==1 && GeneList[0]==tfinput){
		exploretftarget(tfinput);
	} else{
		newWWG = open('','_blank','height=600,width=800,left=1400,top=200,scrollbars=yes');
		newWWG.document.write("<body>matches </body>");
		
		var genediv=d3.select(newWWG.document.body)
			.style("background","white")
			.append("div")
			.style("padding-left","50px")
			.style("padding-top","50px")
			.attr("width",400)
			.attr("height",600);
		createTable(genediv,"mtable",GeneList,exploretftarget);
	}	
}


// explore regulator target
var exploretftarget=function(tfinput,geneplotdiv){
	if (tfinput===undefined){
		var tfinput=d3.select("#tfName").property("value").toUpperCase();
	}
	tfinput=tfinput.toUpperCase();
	var tftargets=regTargetMap[tfinput];
	var targetexList=[];
	for (var target of tftargets){
		var targetIndex=exgene.indexOf(target);
		var targetex=genex[targetIndex];
		targetexList.push(targetex);
	}
	
	var avt=[];
	for (var i=0;i<targetexList[0].length;i++){
		var avi=[];
		for (var target of targetexList){
			avi.push(target[i]);
		}
		var sumi=avi.reduce(function(a,b){return a+b;},0);
		sumi=(sumi/avi.length);
		var ti=timePoints[i];
		avt.push([ti,sumi]);
	}
	
	if (geneplotdiv==undefined){
		var newW2 = open('','_blank','height=800,width=1000,left=1000,top=200,scrollbars=yes');
		newW2.document.write("<head><link rel='stylesheet' type='text/css' href='style.css'></head><body></body>");
		geneplotdiv=d3.select(newW2.document.body).append("div");
	}
	
	gplotData(geneplotdiv,"",avt,"Avg. Expression of "+tfinput+" targets");
}


//wild explore regulator
var wildexploretf=function(tfinput){
	if (tfinput===undefined){
		var tfinput=d3.select("#tfName").property("value").toUpperCase();
	}
	tfinput=tfinput.toUpperCase();
	
	var GeneList=[];
	var localReg=reg.map(function(d){return d.toUpperCase();});
	for(var gi of localReg){
		if (gi.indexOf(tfinput)!=-1){
			GeneList.push([gi]);
		}
	}
	
	if (GeneList.length==1 && GeneList[0]==tfinput){
		exploretf(tfinput);
	} else{
		newWWG = open('','_blank','height=600,width=800,left=1400,top=200,scrollbars=yes');
		newWWG.document.write("<body>matches </body>");
		
		var genediv=d3.select(newWWG.document.body)
			.style("background","white")
			.append("div")
			.style("padding-left","50px")
			.style("padding-top","50px")
			.attr("width",400)
			.attr("height",600)
			.attr("class","div_matched_tfs");
		createTable(genediv,"mtable",GeneList,exploretf);
	}
}


//explore regulator
var exploretf=function(tfinput){
	resetPath();
	if (tfinput===undefined){
		var tfinput=d3.select("#tfName").property("value").toUpperCase();
	}else{
		document.getElementById("tfName").value=tfinput;
	}
	tfinput=tfinput.toUpperCase();
	var paths=d3.select("svg").selectAll("path")[0];
	var tfNodeList=getNodeListReg(tfinput);
	var pathList=getPathListReg(tfinput);
	d3.selectAll(pathList)
	.attr("class","link")
	.attr("stroke","blue");
};

//wild explore mir
var wildexploremir=function(geneinput){
	var GeneList=[];
	
	if (geneinput==undefined){
		var geneinput=d3.select("#mirName").property("value").toUpperCase();
	}else{
		geneinput=geneinput.toUpperCase();
	}

	for(var gi of mirID){
		if (gi.indexOf(geneinput)!=-1){
			GeneList.push([gi]);
		}
	}
	
	if (GeneList.length==1 && GeneList[0]==geneinput){
		exploremir(geneinput);
	} else{
		newWWG = open('','_blank','height=600,width=800,left=1400,top=200,scrollbars=yes');
		newWWG.document.write("<body>matches </body>");
		
		var genediv=d3.select(newWWG.document.body)
			.style("background","white")
			.append("div")
			.style("padding-left","50px")
			.style("padding-top","50px")
			.attr("width",400)
			.attr("height",600)
			.attr("class","div_matched_mirs");
		createTable(genediv,"mtable",GeneList,exploremir);
	}

}


//explore mir
var exploremir=function(mirinput,geneplotdiv){
	if (mirinput===undefined){
		var mirinput=d3.select("#mirName").property("value");
	}
	var X=[];
	var mirIndex;
	mirIndex=mirID.indexOf(mirinput);
	if (mirIndex==-1){
		mirIndex=mirID.map(function(d){return d.toUpperCase()}).indexOf(mirinput.toUpperCase());
	}
	if (mirIndex!=-1){
		var mirEx=mirex[mirIndex];
	}
	for (var i in timePoints){
		X.push([timePoints[i],mirEx[i]]);
	}
	
	if (geneplotdiv==undefined){
		var newW2 = open('','_blank','height=800,width=1000,left=1000,top=200,scrollbars=yes');
		newW2.document.write("<head><link rel='stylesheet' type='text/css' href='style.css'></head><body></body>");
		geneplotdiv=d3.select(newW2.document.body).append("div");
	}
	
	gplotData(geneplotdiv,mirinput,X,"miRNA expression of "+mirinput+" (relative to 0)");
}

//wild explore omnibus 
var wildexploreomnibus=function(geneinput){
	
}

var exploreomnibus=function(geneinput){
	if (geneinput==undefined){
		var geneinput=d3.select("#omnibus-gene").property("value").toUpperCase();	
	}else{
		geneinput=geneinput.toUpperCase();
	}
	
	var newW2 = open('','_blank','height=800,width=1000,left=1000,top=200,scrollbars=yes');
	newW2.document.write("<head><link rel='stylesheet' type='text/css' href='style.css'></head><body></body>");
	var geneplotdiv=d3.select(newW2.document.body).append("div");
	
	// plot gene expression/methylation
	
	if (mirID.indexOf(geneinput)!=-1){
		exploremir(geneinput,geneplotdiv);
	}else{
		exploregene(geneinput,geneplotdiv);
	}
	
	exploreab(geneinput,geneplotdiv);
	exploremethy(geneinput,geneplotdiv);
	
	// plot regulator 
	var localReg=reg.map(function(d){return d.toUpperCase();});
	if (localReg.indexOf(geneinput)!=-1){
		exploretftarget(geneinput,geneplotdiv);
		exploretfmethy(geneinput,geneplotdiv);
		
	}
}

//wild explore absolute expression
var wildexploreab=function(geneinput){
	var GeneList=[];
	if (geneinput==undefined){
		var geneinput=d3.select("#abName").property("value").toUpperCase();	
	}else{
		geneinput=geneinput.toUpperCase();
	}
	
	abx=geneabex.map(function(d){return d[0].toUpperCase();});
	abxm=mirabex.map(function(d){return d[0].toUpperCase();});
	
	abx=abx.concat(abxm);
	
	for(var gi of abx){
		if (gi.indexOf(geneinput)!=-1){
			GeneList.push([gi]);
		}
	}
	
	if (GeneList.length==1 && GeneList[0]==geneinput){
		exploreab(geneinput);
	}else{
		newWWG = open('','_blank','height=600,width=800,left=1400,top=200,scrollbars=yes');
		newWWG.document.write("<body>matches </body>");
		
		var genediv=d3.select(newWWG.document.body)
			.style("background","white")
			.append("div")
			.style("padding-left","50px")
			.style("padding-top","50px")
			.attr("width",400)
			.attr("height",600)
			.attr("class","div_matched_genes");
		createTable(genediv,"mgenetable",GeneList,exploreab);
	}
}

//explore absolute expression
var exploreab=function(geneinput,geneplotdiv){
	if (geneinput==undefined){
		var geneinput=d3.select("#abName").property("value").toUpperCase();	
	}else{
		geneinput=geneinput.toUpperCase();
	}
	abx={}
	abxm={}
	for (var item of geneabex){
		var ex_item=[];
		for (var i=1;i<item.length; i++){
			ex_item.push(parseFloat(item[i]));
		}
		abx[item[0].toUpperCase()]=ex_item;
	}
	
	for (var item of mirabex){
		var ex_item=[];
		for (var i=1;i<item.length; i++){
			ex_item.push(parseFloat(item[i]));
		}
		abxm[item[0].toUpperCase()]=ex_item;
	}
	
	var XX=[];
	if (geneinput in abx){
		var geneX=abx[geneinput];
		for (var i=0;i<timePoints.length;i++){
			XX.push([timePoints[i],geneX[i]]);
		}
	}
	
	if (geneinput in abxm){
		XX=[];
		var geneX=abxm[geneinput];
		for (var i=0;i<timePoints.length;i++){
			XX.push([timePoints[i],geneX[i]]);
		}
	}
	
	if (geneplotdiv==undefined){
		var newW2 = open('','_blank','height=800,width=1000,left=1000,top=200,scrollbars=yes');
		newW2.document.write("<head><link rel='stylesheet' type='text/css' href='style.css'></head><body></body>");
		geneplotdiv=d3.select(newW2.document.body).append("div");
	}
	
	gplotData(geneplotdiv,geneinput,XX,"absolute expression of "+geneinput);
}

// wild search gene 
var wildexploregene=function(geneinput){
	var GeneList=[];
	
	if (geneinput==undefined){
		var geneinput=d3.select("#deName").property("value").toUpperCase();
	}else{
		geneinput=geneinput.toUpperCase();
	}

	for(var gi of exgene){
		if (gi.indexOf(geneinput)!=-1){
			GeneList.push([gi]);
		}
	}
	if (GeneList.length==1 && GeneList[0]==geneinput){
		var newW2 = open('','_blank','height=800,width=1000,left=1000,top=200,scrollbars=yes');
		newW2.document.write("<head><link rel='stylesheet' type='text/css' href='style.css'></head><body></body>");
		geneplotdiv=d3.select(newW2.document.body).append("div");	
		exploregene(geneinput,geneplotdiv);
	}else{
		newWWG = open('','_blank','height=600,width=800,left=1400,top=200,scrollbars=yes');
		newWWG.document.write("<body>matches </body>");
		
		var genediv=d3.select(newWWG.document.body)
			.style("background","white")
			.append("div")
			.style("padding-left","50px")
			.style("padding-top","50px")
			.attr("width",400)
			.attr("height",600)
			.attr("class","div_matched_genes");
		createTable(genediv,"mgenetable",GeneList,exploregene);
	}
}


//explore gene
var exploregene=function(geneinput,geneplotdiv){
	resetPath();
	resetNode();
	
	if (geneinput==undefined){
		var geneinput=d3.select("#deName").property("value").toUpperCase();
	}else{
		geneinput=geneinput.toUpperCase();
	}
	
	var pathList=getPathList(geneinput);
	d3.selectAll(pathList)
	.attr("class","link")
	.attr("stroke","red");
	
	var nodeListSelect=getNodeList(geneinput);
	d3.selectAll(nodeListSelect)
	.append("text")
	.attr("class","nodeExText")
	.text(function(d){
		var geneIndex=exgene.indexOf(geneinput);
		var exVal=(d.nodeMean).toFixed(2);
		return exVal;
		})
	.attr("x",-10)
	.attr("fill","black")
	.style("font-size","10px");
	
	var xgene=[];
	var geneIndex=exgene.indexOf(geneinput);
	var xg=genex[geneIndex];
	for(var t=0;t<timePoints.length;t++){
		xgene.push([timePoints[t],xg[t]]);
	}
	
	if (geneplotdiv==undefined){
		var newW2 = open('','_blank','height=800,width=1000,left=1000,top=200,scrollbars=yes');
		newW2.document.write("<head><link rel='stylesheet' type='text/css' href='style.css'></head><body></body>");
		geneplotdiv=d3.select(newW2.document.body).append("div");	
	}
	
	//plotGene(geneinput,nodeListSelect);
	gplotData(geneplotdiv,geneinput,xgene,"expression of "+geneinput);
};

//explore prot

var exploreprote=function(geneinput,geneplotdiv){
	var GeneList=[];
	
	if (geneinput==undefined){
		var geneinput=d3.select("#proteText").property("value").toUpperCase();
	}else{
		geneinput=geneinput.toUpperCase();
	}
	
	var allprots=protex.map(function(d){
		return d[0].toUpperCase();
	});

	if (geneplotdiv==undefined){
		var newW2 = open('','_blank','height=800,width=1000,left=1000,top=200,scrollbars=yes');
		newW2.document.write("<head><link rel='stylesheet' type='text/css' href='style.css'></head><body></body>");
		geneplotdiv=d3.select(newW2.document.body).append("div");	
	}
		

	
	var geneIndex=allprots.indexOf(geneinput);
	if (geneIndex!=-1){
		var FR=protex[0].slice(1);
		var EX=protex[geneIndex].slice(1);
		var xgene=[];
		for (var i in FR){
			xgene.push([FR[i],parseFloat(EX[i])]);
		}	
		//plotGene(geneinput,nodeListSelect);
		gplotData(geneplotdiv,geneinput,xgene,"abosulte protein level of "+geneinput);		
	} else{
		geneplotdiv.
		html(geneinput+" not found in proteomics data!");
	}
	
}



/*global function section*/
//p-value TF methylation impact

var pvTFMethyImpact=function(tfinput,node){
	var tftargets=regTargetMap[tfinput];
	var methytime=timePoints[Math.max(0,node.depth-1)];
	var methygenes;
	if (methytime in methyscore){
		methygenes=methyscore[methytime];
	}else{
		return ["no_methylation_data","no_methylation_data","no_methylation_data"];
	}
	
	var genes_node_from=node.parent.genes;
	var genes_node=node.genes;
	
	var targets_node_from=[];
	for (var target of tftargets){
		if (genes_node_from.indexOf(target)!=-1){
			targets_node_from.push(target);
		}
	}
	var targets_node=[];
	for (var target of tftargets){
		if (genes_node.indexOf(target)!=-1){
			targets_node.push(target);
		}
	}
	
	var methycut=0.5
	var targets_node_from_methy=[];
	for (var target of targets_node_from){
		if (target in methygenes){
			if (methygenes[target]>methycut){
				targets_node_from_methy.push(target);
			}
		}
	}
	
	var targets_node_methy=[];
	for (var target of targets_node){
		if (target in methygenes){
			if (methygenes[target]>methycut){
				targets_node_methy.push(target);
			}
		}
	}
	
	var prob_bg=genes_node.length*1.0/genes_node_from.length;
	if (node.parent.children.length>1){
		var pv=1-pbinom(targets_node_methy.length-1,targets_node_from_methy.length,prob_bg);
		var pv2=1-pbinom(targets_node.length-1,targets_node_from.length,prob_bg);
		var sp1=targets_node_methy.length*1.0/targets_node_from_methy.length*100;
		var sp2=targets_node.length*1.0/targets_node_from.length*100;
		var pv=pv.toExponential(3);
		var pv2=pv2.toExponential(3);
		var sp1=sp1.toFixed(2);
		var sp2=sp2.toFixed(2);
	}else{
		var pv='No split';
		var pv2='No split';
		var sp1='No split';
		var sp2='No split';
	}
	return [node.nodeID,tftargets.length,targets_node_from.length,targets_node_from_methy.length,targets_node.length,targets_node_methy.length,pv,pv2,sp1,sp2];
};


//methylation data for input gene

var ScoreMethyGene=function(geneinput){
	geneinput=geneinput.toUpperCase();
	var ScoreList=[];
	for (var time of timePoints){
		if (time in methyscore){ 
			if (geneinput in methyscore[time]){
				var methyValue=methyscore[time][geneinput];
				ScoreList.push([time,methyValue]);
			//}else{
				//ScoreList.push([time,0]);
			}		
		}
	}
	return ScoreList;
};

// plot 2d heatmap

var plotHeatMap=function(X,plotitle){
		var newW2 = open('','_blank','height=700,width=600,left=1000,top=200,scrollbars=yes')
		newW2.document.write("<head><title>"+plotitle+ "</title><link rel='stylesheet' type='text/css' href='style.css'></head><body></body>");
		
		var height=600;
		var width=200;
		var rows=X[0].length;
		var cols=X.length;		
		var row_height=height*1.0/X.length;
		var col_width=width*1.0/X[0].length;
		
		var chart=d3.select(newW2.document.body).append("div")
					.attr("class","plotheatmap")
					.append("svg")
					.attr("width",200-col_width/2)
					.attr("height",700)
					.style("margin-left","10px")
					.style("margin-right","10px")
					.style("padding-right","10px")
					.style("border","1px solid")
					.style("background","silver");
						
		var ymin=0;
		var ymax=0;
		for (var xx of X){
			for (var xxy of xx){
				if (ymin>xxy){
					ymin=xxy;
				}
				if (ymax<xxy){
					ymax=xxy;
				}
			}
		}
	
	
		var x = d3.scale.ordinal()
			.rangeRoundBands([0, width], .1);
		var xAxis = d3.svg.axis()
		.scale(x)
		.orient("bottom");
		
		
			
		var yoffset=0;
		var color=d3.scale.linear();
		color.domain([ymin,ymax]);
		color.range(["white","red"]);		
	
		var methytimes=[];
		for (var time of timePoints){
			if (time in methyscore){
				methytimes.push(time);
			}
		}
		
		x.domain(methytimes);
		var offset=100;
		for (var i in X){
			chart.append("g")
				.selectAll("rect")
				.data(X[i])
				.enter()
				.append("rect")
				.attr("x",function(d,j) {return col_width*j;})
				.attr("y",function(d,j) {return i*row_height;})
				.attr("width",col_width/2)
				.attr("height",row_height)
				.attr("fill",function(d){return color(d);});
		}
		
		chart.append("g")
		  .attr("class", "x axis")
		  .attr("transform", "translate(0," + (height+yoffset) + ")")
		  .call(xAxis)
		  .selectAll("text")
		  .attr("transform","rotate(90)")
		  .style("stroke","black")
		  .attr("y",0)
		  .attr("x",6)
		  .attr("fill","red")
		  .style("text-anchor", "start")
		  .style("text-color","black");
		
					
}

// plot gene expression 
/*
var plotGene=function(geneinput,selectNodeList){
		var newW2 = open('','_blank','height=600,width=800,left=1400,top=200,scrollbars=yes');
		newW2.document.write("<head><title>Plot Gene</title> <link rel='stylesheet' type='text/css' href='style.css'></head><body></body>");
		
		var chart=d3.select(newW2.document.body).append("div")
					.attr("class","plottf")
					.append("svg")
					.attr("fill","black")
					.attr("width",800)
					.attr("height",700)
					//.style("border","1px solid")
					.style("margin-left","10px")
					.style("margin-right","10px")
					.style("padding-right","10px")
					.style("border","1px solid");
					
		var width = 600;
		var height = 200;
		var yoffset=300;
		
		var x = d3.scale.ordinal()
			.rangeRoundBands([0, width], .1);

		var y = d3.scale.linear()
			.range([height, 0]);
		
		var xAxis = d3.svg.axis()
		.scale(x)
		.orient("bottom");
		
		
		var yAxis = d3.svg.axis()
			.scale(y)
			.orient("left")
			.ticks(10);
			
		
		var data=[];
		for (var snode of selectNodeList){
			data.push(snode.__data__);
		}
		
		var geneIndex=exgene.indexOf(geneinput);
		var exVal;
		
		x.domain(data.map(function(d) { return d.nodetime; }));
		var ymax=0;
		var ymin=0;
		
		for (var node of nodes){
			if (ymax<node.nodeMean){
				ymax=node.nodeMean;
			}
			
			if (ymin>node.nodeMean){
				ymin=node.nodeMean;
			}
		}
		
		y.domain([-6.5,6.5]);
	
		
		var bar = chart.selectAll("g")
		  .data(data)
		  .enter().append("g")
		  .attr("transform", function(d) { return "translate(" + (x(d.nodetime)+100) + ","+(yoffset)+")"; });

		bar.append("rect")
		  .attr("y", function(d) { 
			  exVal=genex[geneIndex][d.depth].toFixed(2);
			  if (exVal>0){
				  var yval=y(exVal);
			  }else{
				  var yval=y(0);
			  }
			  return yval; })
		  .attr("height", function(d) {
			  exVal=genex[geneIndex][d.depth].toFixed(2);
			  return Math.abs(y(exVal)-y(0)); })
		  .attr("width", x.rangeBand())
		  .attr("fill","steelblue");

		bar.append("text")
		  .attr("x", x.rangeBand() / 2-20)
		  //.attr("y",100)
		  .attr("y", function(d) { 
			  exVal=genex[geneIndex][d.depth].toFixed(2);
			  if (exVal>0){
				 var yval=y(0)-Math.abs(y(exVal)-y(0))-20;
			  }else{
				  var yval=y(exVal)+10;
			  }
			  return yval; })
		  .attr("dy", ".75em")
		  .text(function(d) { 
			  exVal=genex[geneIndex][d.depth].toFixed(2);
			  return exVal; });
		
		chart.append("g")
		  .attr("class", "x axis")
		  .attr("transform", "translate(100," + (height+yoffset) + ")")
		  .call(xAxis)
		  .selectAll("text")
		  .attr("transform","rotate(90)")
		  .style("stroke","black")
		  .attr("y",0)
		  .attr("x",6)
		  .attr("fill","black")
		  .style("text-anchor", "start");
		  //attr("dy","3em")


		chart.append("g")
		  .attr("class", "y axis")
		  .attr("transform", "translate(100," + (0+yoffset) + ")")
		  .call(yAxis)
		  .append("text")
		  .style("stroke","black")
		  .attr("dx","0em")
		  .attr("y",50)
		  .attr("transform","rotate(90)")
		  .attr("dy","0.71em")
		  .attr("fill","black")
		  .style("text-anchor","start")
		  .text("Relative Expression (to time 0) of "+geneinput);

}
*/
var plotGene=function(geneinput,selectNodeList){
		var newW2 = open('','_blank','height=600,width=800,left=1400,top=200,scrollbars=yes');
		newW2.document.write("<head><title>Plot Gene</title> <link rel='stylesheet' type='text/css' href='style.css'></head><body></body>");
		
		var genechart=d3.select(newW2.document.body).append("div")
					.attr("id","geneplot")
					
		var data=[];
		for (var snode of selectNodeList){
			data.push(snode.__data__);
		}
		
		var geneIndex=exgene.indexOf(geneinput);
		var exVal;
		
		x.domain(data.map(function(d) { return d.nodetime; }));
		var ymax=0;
		var ymin=0;
		
		for (var node of nodes){
			if (ymax<node.nodeMean){
				ymax=node.nodeMean;
			}
			
			if (ymin>node.nodeMean){
				ymin=node.nodeMean;
			}
		}
		
		y.domain([-6.5,6.5]);
	
		
		var bar = chart.selectAll("g")
		  .data(data)
		  .enter().append("g")
		  .attr("transform", function(d) { return "translate(" + (x(d.nodetime)+100) + ","+(yoffset)+")"; });

		bar.append("rect")
		  .attr("y", function(d) { 
			  exVal=genex[geneIndex][d.depth].toFixed(2);
			  if (exVal>0){
				  var yval=y(exVal);
			  }else{
				  var yval=y(0);
			  }
			  return yval; })
		  .attr("height", function(d) {
			  exVal=genex[geneIndex][d.depth].toFixed(2);
			  return Math.abs(y(exVal)-y(0)); })
		  .attr("width", x.rangeBand())
		  .attr("fill","steelblue");

		bar.append("text")
		  .attr("x", x.rangeBand() / 2-20)
		  //.attr("y",100)
		  .attr("y", function(d) { 
			  exVal=genex[geneIndex][d.depth].toFixed(2);
			  if (exVal>0){
				 var yval=y(0)-Math.abs(y(exVal)-y(0))-20;
			  }else{
				  var yval=y(exVal)+10;
			  }
			  return yval; })
		  .attr("dy", ".75em")
		  .text(function(d) { 
			  exVal=genex[geneIndex][d.depth].toFixed(2);
			  return exVal; });
		
		chart.append("g")
		  .attr("class", "x axis")
		  .attr("transform", "translate(100," + (height+yoffset) + ")")
		  .call(xAxis)
		  .selectAll("text")
		  .attr("transform","rotate(90)")
		  .style("stroke","black")
		  .attr("y",0)
		  .attr("x",6)
		  .attr("fill","black")
		  .style("text-anchor", "start");
		  //attr("dy","3em")


		chart.append("g")
		  .attr("class", "y axis")
		  .attr("transform", "translate(100," + (0+yoffset) + ")")
		  .call(yAxis)
		  .append("text")
		  .style("stroke","black")
		  .attr("dx","0em")
		  .attr("y",50)
		  .attr("transform","rotate(90)")
		  .attr("dy","0.71em")
		  .attr("fill","black")
		  .style("text-anchor","start")
		  .text("Relative Expression (to time 0) of "+geneinput);

}


//line plot
var gplotData=function(container,inputgene,x,ytitle,xcols){
	
	Keys=['LineChart','BarChart','ColumnChart'];
	
	var plotypediv=container.append("div")
					.append("select")
					.attr("id","plotype")
					.on("change",function(){
					var selectType=this.options[this.selectedIndex].value;
					if (selectType=="LineChart"){
							gtypeplot(chartdiv,xdata,options,'LineChart');
					}
					if (selectType=="BarChart"){
							gtypeplot(chartdiv,xdata,options,'BarChart');
					}
					if (selectType=="ColumnChart"){
							gtypeplot(chartdiv,xdata,options,"ColumnChart");
					}
					
					})
					.selectAll("option")
					.data(Keys)
					.enter()
					.append("option")
					.text(function(d){
						return d;
					})
					.attr("value",function(d){
						return d;
						});
	
	var chartdiv=container.append("div")
					.attr("id",inputgene)[0][0];
					
    var xdata=new google.visualization.DataTable();
    xdata.addColumn('string', 'Time');
    if (xcols!=undefined){
		for (var ci=0;ci<xcols.length;ci++){
			xdata.addColumn('number',xcols[ci]);
		}
	}else{
		xdata.addColumn('number', 'expression');
	}
	
	xdata.addRows(x);
    var options={
		 title: ytitle,
		 width: 800,
		 height: 600,
		 legend: {position: 'bottom'}
		 };
	gtypeplot(chartdiv,xdata,options,'LineChart');
   
}
// google charts with different types of charts
var gtypeplot=function(chartdiv,xdata,options,typePlot){
	if (typePlot=='ColumnChart'){
		var chart=new google.visualization.ColumnChart(chartdiv);
	}else if (typePlot=='BarChart'){
		var chart=new google.visualization.BarChart(chartdiv);
	}else{
		var chart=new google.visualization.LineChart(chartdiv);
	}
    chart.draw(xdata,options);
}


//plot data
var plotData=function(inputgene,X,ytitle){
	// X[0] -> x axis
	// X[1] -> y axis
	
	var newW2 = open('','_blank','height=800,width=800,left=1000,top=200,scrollbars=yes');
	newW2.document.write("<head><link rel='stylesheet' type='text/css' href='style.css'></head><body></body>");
		
	var chart=d3.select(newW2.document.body).append("div")
					.append("svg")
					.attr("fill","black")
					.attr("width",800)
					.attr("height",800)
					//.style("border","1px solid")
					.style("margin-left","10px")
					.style("margin-right","10px")
					.style("padding-right","10px")
					.style("border","1px solid");
					
	var width = 600;
	var height = 200;
	var yoffset=300;
	
	var x = d3.scale.ordinal()
		.rangeRoundBands([0, width], .1);

	var y = d3.scale.linear()
		.range([height, 0]);
	
	var xAxis = d3.svg.axis()
	.scale(x)
	.orient("bottom");
	
	
	var yAxis = d3.svg.axis()
		.scale(y)
		.orient("left")
		.ticks(10);
	
	x.domain(X.map(function(d) { return d[0]; }));
	var ymin=0;
	var ymax=0;
	for (var xx of X){
		if (ymin>xx[1]){
			ymin=xx[1];
		}
		if (ymax<xx[1]){
			ymax=xx[1];
		}
	}
	
	y.domain([ymin-0.1,ymax+0.1]);
	
	
	var bar = chart.selectAll("g")
		  .data(X)
		  .enter().append("g")
		  .attr("transform", function(d) { return "translate(" + (x(d[0])+100) + ","+(yoffset)+")"; });
	
	var xrangeband=x.rangeBand();
	
	bar.append("rect")
	  .attr("y", function(d) { 
		  exVal=d[1].toFixed(2);
		  if (exVal>0){
			  var yval=y(exVal);
		  }else{
			  var yval=y(0);
		  }
		  return yval; })
	  .attr("height", function(d) {
		  exVal=d[1].toFixed(2);
		  return Math.abs(y(exVal)-y(0)); })
	  .attr("width", xrangeband)
	  .attr("fill","steelblue");

	bar.append("text")
	  .attr("x", xrangeband / 2-20)
	  //.attr("y",100)
	  .attr("y", function(d) { 
		  exVal=d[1].toFixed(2);
		  if (exVal>0){
			 var yval=y(0)-Math.abs(y(exVal)-y(0))-20;
		  }else{
			  var yval=y(exVal)+10;
		  }
		  return yval; })
	  .attr("dy", ".75em")
	  .text(function(d) { 
		  exVal=d[1].toFixed(2);
		  return exVal; });
	
	chart.append("g")
	  .attr("class", "x axis")
	  .attr("transform", "translate(100," + (height+yoffset) + ")")
	  .call(xAxis)
	  .selectAll("text")
	  .attr("transform","rotate(90)")
	  .style("stroke","black")
	  .attr("y",0)
	  .attr("x",6)
	  .attr("fill","black")
	  .style("text-anchor", "start")
	  .attr("dx","1em");


	chart.append("g")
	  .attr("class", "y axis")
	  .attr("transform", "translate(100," + (0+yoffset) + ")")
	  .call(yAxis)
	  .append("text")
	  .style("stroke","black")
	  .attr("dx","0em")
	  .attr("y",50)
	  .attr("transform","rotate(90)")
	  .attr("dy","0.71em")
	  .attr("fill","black")
	  .style("text-anchor","start")
	  .text(ytitle+' '+inputgene);
	
}

//get node list for a regulator 

var getNodeListReg=function(reg){
	var AllNodeList=d3.select("svg").selectAll("g.node")[0];
	var selectNodeList=[];
	var tfcut=document.getElementById("tfcutSelector").innerHTML;
	tfcut=parseInt(tfcut);
	
	for (var node of AllNodeList){
		var node_regs=node.__data__.ETF;
		var regList=[];
		if (node_regs){
			for (var i=0;i<tfcut;i++){
				regList.push(node_regs[i][0].split(" ")[0].toUpperCase());
			}
		}
		if (regList.indexOf(reg)!=-1){
			selectNodeList.push(node);
		}
	}
	return selectNodeList;
};


//get path list for a regulator

var getPathListReg=function(reg){
	var tfcut=document.getElementById("tfcutSelector").innerHTML;
	tfcut=parseInt(tfcut);
	var paths=d3.select("svg").selectAll("path")[0];
	var pathList=[];
	for (var path of paths){
		var target=path.__data__.target;
		var node_regs=target.ETF;
		var regList=[];
		if (node_regs){
			for (var i=0;i<tfcut;i++){
				regList.push(node_regs[i][0].split(" ")[0].toUpperCase());
			}
		}
		if (regList.indexOf(reg)!=-1){
			pathList.push(path);
		}
	}
	return pathList;
}

//get node list for a gene
var getNodeList=function(gene){
	var AllnodeList=d3.select("svg").selectAll("g.node")[0];
	var selectNodeList=[];
	for (var node of AllnodeList){
		node_genes=node.__data__.genes;
		if (node_genes.indexOf(gene)!=-1){
			selectNodeList.push(node);
		}
		if (node.__data__.nodeID==0){
			selectNodeList.push(node);
		}
	}
	
	return  selectNodeList;
}


//get path List for a gene
var getPathList=function(gene){
	var paths=d3.select("svg").selectAll("path")[0];
	var pathList=[];
	for (var path of paths){
		var target=path.__data__.target;
		var target_genes=target.genes;
		if (target_genes.indexOf(gene)!=-1){
			pathList.push(path);
		}
	}
	return pathList;
};

//create a dropdown menu

var createDropDown=function(FirstRow,tfdropdowndiv,dropdownid,TFs,onChange){
	var Keys=[FirstRow];
	TFs.sort();
	for (var tf of TFs){
		Keys.push(tf);
	}
	
	d3.select(tfdropdowndiv)
	.select("select")
	.remove();
	
	d3.select(tfdropdowndiv)
	.append("select")
	.attr("id",dropdownid)
	.on("change",onChange)
	.selectAll("option")
	.data(Keys)
	.enter()
	.append("option")
	.text(function(d){
		return d;
	})
	.attr("value",function(d){
		return d;
		});
	
}

//append a table to the cant
var createCompareTable=function(cant,tableid,data){
	var tfCount={};
	for (var row of data){
		for (var rowcol of row){
			if ((rowcol in tfCount)==false){
				tfCount[rowcol]=1;
			}else{
				tfCount[rowcol]+=1;
			}
		}
	}
	
	cant.append("table")
	.attr("id",tableid)
	.style("border","1px solid")
	.style("border-collapse","collapse")
	.selectAll("tr")
	.data(data)
	.enter()
	.append("tr")
	.style("border","1px solid")
	.selectAll("td")
	.data(function(d){return d;})
	.enter()
	.append("td")
	.style("border","1px solid")
	.text(function(d){return d;})
	.on("click",function(d){
		exploregene(d);
		})
	.style("background-color",function(d){
			if (tfCount[d]==2){
				return "orange";
			}else if (tfCount[d]==3){
				return "red";
			}
		});
}


//append a table to the cant
var createTFTable=function(cant,tableid,data,responseFunction,depth){
	cant.append("table")
	.attr("id",tableid)
	.style("border","1px solid")
	.style("border-collapse","collapse")
	.selectAll("tr")
	.data(data)
	.enter()
	.append("tr")
	.style("border","1px solid transparent")
	.style("height","30px")
	.style("transition", "all 0.3s")
	.style("background", function(d,i){
		if (i==0){
			return "#DFDFDF";
		}else if (i%2==0){
			return "#F1F1F1";
		} return "#FEFEFE";
		
	})
	.style("font-weight",function(d,i){
		if (i==0){
			return "bold";
		}
	})
	.style("color",function(d){
			var geneIndex=exgene.indexOf(d[0].split(" ")[0].toUpperCase());
			if (geneIndex==-1){
				geneIndex=mirID.indexOf(d[0].split(" ")[0].toUpperCase());
				var gvalue=mirex[geneIndex];
			}else{
				var gvalue=genex[geneIndex];
			}
			if (gvalue!=undefined){
				gvalue=gvalue[depth];
			}
			
			if (gvalue>0){
				return "blue";
			}else if (gvalue<0){
				return "red";
			}else if (gvalue==0){
				return "silver";
			}else{
				return "gray";
			}
		})
	.selectAll("td")
	.data(function(d){return d;})
	.enter()
	.append("td")
	.style("border","1px dotted")
	.text(function(d){return d;})
	.on("click",function(d){
			var key=d.split(" ")[0];
			if(mirID.indexOf(key)!=-1){
				exploremir(key);
			}else{
				exploregene(key);
			}
		});
	return cant;
};


//append a table to the cant
var createTable=function(cant,tableid,data,responseFunction){
	cant.append("table")
	.attr("id",tableid)
	//.style("border","1px solid")
	.style("border-collapse","collapse")
	.style("color","#333")
	.style("font-family","Helvetica, Arial, sans-serif")
	.style("border-spacing","0")
	.selectAll("tr")
	.data(data)
	.enter()
	.append("tr")
	.style("border","1px solid transparent")
	.style("height","30px")
	.style("transition", "all 0.3s")
	.style("background", function(d,i){
		if (i==0){
			return "#DFDFDF";
		}else if (i%2==0){
			return "#F1F1F1";
		} return "#FEFEFE";
		
	})
	.style("font-weight",function(d,i){
		if (i==0){
			return "bold";
		}
	})
	.selectAll("td")
	.data(function(d){return d;})
	.enter()
	.append("td")
	.style("border","1px dotted")
	.text(function(d){return d;})
	.on("click",function(d){
			responseFunction(d);
		});
	return cant;
};


//Math functions
var erf=function(z){
	var t = 1.0 / (1.0 + 0.5 * Math.abs(z));
	var ans = 1 - t * Math.exp( -z*z -  1.26551223 +
											t * ( 1.00002368 +
											t * ( 0.37409196 + 
											t * ( 0.09678418 + 
											t * (-0.18628806 + 
											t * ( 0.27886807 + 
											t * (-1.13520398 + 
											t * ( 1.48851587 + 
											t * (-0.82215223 + 
											t * ( 0.17087277))))))))));
	if (z > 0.0){
			return ans;
	}else{
			return -ans
	}

}

//pbinom
var pbinom=function(s, n, p){
    var u = n * p;
    var o=Math.pow(u*(1-p),0.5);
    //var out=0.5*(1+erf((s-u)/(Math.pow(o*2,0.5))));
    var out=0.5*(1+erf((s-u)/(o*Math.pow(2,0.5))));
    return out;
}

//Gaussian probability

var NormProb=function(x,mean,standardDeviation){
	if (standardDeviation==0){
		return 0;
	}
    return -1*Math.log(Math.pow(2*Math.PI*standardDeviation * standardDeviation,-0.5)*Math.pow(Math.E, -Math.pow(x - mean, 2) / (2 * (standardDeviation * standardDeviation))));
}

//nodeCompare
var nodeCompare=function(nodeA,nodeB){
	
	if (nodeA.nodeMean<nodeB.nodeMean){
		return 1;
	}else if (nodeA.nodeMean>nodeB.nodeMean){
		return -1;
	}else{
		return 0;
	}
}

//

var createjsondownload=function(jsondownloadlinkid){
	 //var dataObj=JSON.parse(data);
	 //var outObj={"GeneList":dataObj[0], "CellList":dataObj[1], "NodeList":dataObj[2],"EdgeList":dataObj[3]};
	 var rawObj=data[0];
	 var outObj=[];
	 for (var e of rawObj){
		 var ue={"id":e["id"],"nodetime":e["nodetime"],"nodeMean":e["nodeMean"],"nodeSigma":e["nodeSigma"],"ETF":e["ETF"],"genes":e["genes"]};
		 outObj.push(ue);
	 }
	 var outObjStr=JSON.stringify(outObj,null,"	");
	 var blob=new Blob([outObjStr],{type:'application/json;charset=utf-8'});
	 outurl=window.URL.createObjectURL(blob);
	 d3.select("#"+jsondownloadlinkid)
	.attr("href",outurl)
	.attr("download","download.json")
	.text("Ready,Click to download");
}
	
var downloadjson=function(jsondownloadlinkid){
			plswait(jsondownloadlinkid);
			window.setTimeout(function(){createjsondownload(jsondownloadlinkid);},10);
}

function plswait(id){
		document.getElementById(id).innerHTML="wait...";
		d3.select("#"+id)
		.text("Generating file,please wait...");	
}

