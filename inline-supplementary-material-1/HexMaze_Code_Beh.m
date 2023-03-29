%% Code Providing a Demonstration of the behavioral analyis presented in the paper 
% "Learning-induced shifts in mice navigational strategies are unveiled by a minimal behavioral model of spatial exploration"
% Here we compute Relative Trial Length and Distance from Optimal Path over different groups of trials.  
% Example application to HexMaze Build-up Session 1 is provided. 


clear all
close all

%% PREPARE GRAPH


% Node-to-Node Transition Defining HexMaze structure 

Transition_Map = [6 2 0;
                  1 3 0   
                  2 7 4
                  3 0 5
                  4 0 8
                  1 9 17
                  3 9 10
                  5 10 11
                  6 7 12
                  7 8 13
                  8 14 0
                  19 15 9
                  10 15 16
                  11 16 0
                  12 22 13
                  13 24 14
                  6 18 0
                  17 19 0
                  18 12 20
                  0 19 21
                  0 20 22
                  15 21 23
                  0 22 24
                  16 23 0];


% Classify nodes based on internal-external ring of the HexMaze
Border_Map= ones(24,1)*2;
Border_Map([7 9 10 12 13 15])=1;
              
% Build the graph representation of the HexMaze     
s=[];
t=[];
for node=1:size(Transition_Map,1)
    add_n=Transition_Map(node,:);
    add_n=add_n(add_n>0);
    for aa=1:numel(add_n)
        s=cat(2,s,node);
        t=cat(2,t,add_n(aa));
        
    end
    



end


G_Base=graph(s,t);           


% Plot the Graph
G_Dir = digraph(s,t);
figure(2)
plot(G_Dir,'LineWidth',3,'NodeFontSize',20,'NodeFontWeight','bold','EdgeAlpha',0.6,'MarkerSize',10,'ArrowSize',15)


%%
% Load Data
load('HexMazeExData.mat','Trial_n','Start_Loc','Goal_Loc','Barr','Ratio','P_Trials')

% Trial_n : trial label 
% Start_Loc : vector with list of starting node per trial 
% Goal_Loc : vector with list of target node per trial 
% Barr: string array conting code for barrier configuration (if applicable)
% Ratio : array contining Relative Trial Length per trial (can be computed independently from the rest of the variables)
% P_Trials : list of nodes visited in each trial


%% 




% Storage Variables for Curve Fit and for Direct Path Probability 
Direct_P = zeros(1,4); 
Fits_Para=[];




% Cycle Trial Group         
for tri=1:4

% Select Trial Group         
if(tri==1)
tr_ch = Trial_n<=1 & Trial_n>0; % First Trial only
else
        
tr_ch = Trial_n<=1+10*(tri-1) & Trial_n>1+(tri-2)*10; % Groups of 10 trials 
end


% Get Properties for selected Trials

Start_List=Start_Loc(find(tr_ch),1);
Goal_List=Goal_Loc(find(tr_ch),1);

barriers = Barr(find(tr_ch),1);        

OP_Ratio_mice=Ratio(find(tr_ch),1);  
 
         
n_trials=size(Start_List,1);    



Paths=P_Trials(find(tr_ch),1); %paths per trials in group of 10s
Paths=table2cell(Paths); %Convert table to cells
Paths = rmmissing(Paths);

% Store for Distance from Optimal Path
Dist_OP={};


% Shuffled Goal List for Randomized Data
Goal_List_R = Goal_List(randperm(numel(Goal_List)));
OP_Ratio_mice_n = zeros(size(OP_Ratio_mice)); 

 for tr=1:n_trials        %create random starting-ending locations     

  St_Node=Start_List(tr);
  En_Node=Goal_List(tr);
 
% Check Barrier Configuration 
    if barriers(tr,1)=="bar1"
    Barrier_Position = [17 18; 11 14; 15 22];
    elseif barriers(tr,1)=="bar2"
    Barrier_Position = [4 5; 7 9; 20 21];  
    elseif barriers(tr,1)=="bar3" 
    Barrier_Position = [1 2; 13 15; 23 24];
    elseif barriers(tr,1)=="bar4" 
    Barrier_Position = [7 10; 8 11; 18 19];
    elseif barriers(tr,1)=="bar5" 
    Barrier_Position = [3 4; 12 15; 16 24];
    elseif barriers(tr,1)=="bar6" 
    Barrier_Position = [5 8; 9 12; 21 22];
    elseif barriers(tr,1)=="bar7" 
    Barrier_Position = [6 9; 13 16; 19 20];
    elseif barriers(tr,1)=="bar8" 
    Barrier_Position = [2 3; 10 13; 22 23];
    elseif barriers(tr,1)=="bar9" 
    Barrier_Position = [3 7; 6 17; 14 16];
    elseif barriers(tr,1)=="bar10" 
    Barrier_Position = [1 6; 8 10; 15 22];
    elseif barriers(tr,1)=="bar11" 
    Barrier_Position = [7 9; 12 19; 23 24];
    elseif barriers(tr,1)=="bar12" 
    Barrier_Position = [1 2; 12 15; 11 14];
    elseif barriers(tr,1)=="bar13" 
    Barrier_Position = [17 19; 13 15; 8 11];
    elseif barriers(tr,1)=="bar14" 
    Barrier_Position = [1 2; 9 12; 11 14];
else
     Barrier_Position = [];   
    end 
    
% Update Transition Matrix to Include the eventual effect of Barriers

Transition_Map_Barrier = Transition_Map;
for bb=1:size(Barrier_Position,1)
    
  To_Eliminate=find(Transition_Map_Barrier(Barrier_Position(bb,1),:)==Barrier_Position(bb,2));
  Transition_Map_Barrier(Barrier_Position(bb,1),To_Eliminate)=0;
  To_Eliminate=find(Transition_Map_Barrier(Barrier_Position(bb,2),:)==Barrier_Position(bb,1));
  Transition_Map_Barrier(Barrier_Position(bb,2),To_Eliminate)=0;
end
 
% Build the graph representation of the maze

s=[];
t=[];
for node=1:size(Transition_Map_Barrier,1)
  add_n=Transition_Map_Barrier(node,:);
  add_n=add_n(add_n>0);
  for aa=1:numel(add_n)
    s=cat(2,s,node);
    t=cat(2,t,add_n(aa));
  end
end
G_barrier=graph(s,t); % Graph Object


% Compute all possible optimal paths given the initial and goal locations 

    opt_path_cum=[];
    for rr=1:20
    ww=ones(size(s));
    ww=ww+rand(size(ww))*0.01; %randomly adds weight to some nodes so it will take all shortest paths randomly
        
    GG_Noise=digraph(s,t,ww);
    [opt_path,opt_d]=shortestpath(GG_Noise,St_Node,En_Node); %finds the optimal path and distance for the st-end node of each path
    opt_path_cum=cat(1,opt_path_cum,opt_path(:));
    
    end
    opt_path=unique(opt_path_cum); % Contains all nodes belonging to any of the available optimal paths
    
   
    
    vec=str2num(Paths{tr}); % Take current trial node sequence
    
    if ismember(En_Node,vec)
        L = find(vec==En_Node,1,'first');
        
        OP_Ratio_mice_n(tr) = L/opt_d;
    
    else
        OP_Ratio_mice_n(tr) = 50;
    end
    
    
    dist_m=ones(numel(vec),1)*100; % Vector containing step-by-step distance from the optimal path
    ts=0;
    for step=1:length(vec)
        
          ts=ts+1;
        for nn=1:numel(opt_path)
        [~,dist] = shortestpath(G_barrier,vec(step),opt_path(nn));
        if(dist<dist_m(step))
        dist_m(ts)=dist;
         end
        end
       
    end
      
    
    
    dist_m(dist_m==100)=[];
    
     Dist_OP{tr} = dist_m;
    
    
    
    

 end


 
 


 Grand_mean=zeros(50,1);
 Grand_mean_c=zeros(50,1);
 kk=0;
 

 
 for tt=1:numel(Dist_OP)
    if(numel(Dist_OP{tt})<50)
    kk=kk+1;
        Grand_mean(1:numel(Dist_OP{tt}))=Grand_mean(1:numel(Dist_OP{tt}))+Dist_OP{tt};%./D_Normalization;
        Grand_mean_c(:)=Grand_mean_c(:)+1;%./D_Normalization;
    
    end
     
     
 
 
 end
 
 Grand_mean = Grand_mean./Grand_mean_c;
 


 % Plot Distance From Optimal Path
figure(1)

hold on
 plot(Grand_mean,'LineWidth',2,'Color',[1-tri/4 0 tri/4])
xlabel('Node choices')
ylabel('Mean')
ylim([0 3])
xlim([0 50]) %only for the scaled paths


title(['Session ', num2str(1), ' Build-up '])


x_label = (1:size(Grand_mean,1))';

% Parametric Fit to Distance from Optimal Path
y = Grand_mean;
   
F_Fit = @(x,xdata)x(1)*1/(exp(-2*log(x(3)/x(2))/(1-(x(2)/x(3)).^2))-exp(-2*log(x(3)/x(2))/((x(3)/x(2)).^2-1)))*(exp(-xdata.^2/(2*x(2).^2))-exp(-xdata.^2/(2*x(3).^2)));

x0 = [1.5 5 1];
lb = [0 0 0];
ub = [Inf Inf Inf];
options = optimoptions('lsqcurvefit','Algorithm','levenberg-marquardt');
[x] = lsqcurvefit(F_Fit,x0,x_label,y,lb,ub,options);
Fits_Para=cat(1,Fits_Para,x);

% Plot Fit
figure(2)
plot(y,'r')
hold on
plot(F_Fit(x,x_label),'b')
title(['Session', num2str(1)])
hold on



% Plot Relative Trial Length

figure(3)
HH = histcounts(OP_Ratio_mice_n,1:0.5:5,'Normalization','probability');

subplot(1,2,1)
plot(1:0.5:4.5,HH,'LineWidth',2,'Color',[1-tri/4 0 tri/4])
xlabel('Relative Trial Length')
ylabel('Probability')
title(['DTSP Session ' num2str(1)])
hold on

% Take Probability of Low-RTL Trials
Direct_P(1,tri) = HH(1);



end
     

     
     figure(1 )
 subplot(1,2,1)
 legend('Trial 1','Trials 2-11','Trials 12-21','Trials 22-31')
     
    % Plot Probability of Low-RTL along session  

  figure(3)
  subplot(1,2,2)
  for sess=1:1
plot(Direct_P(1,:),'LineWidth',2,'Color',[1-sess/4 0 sess/4])
hold on
  end
xlabel('Trial Group')
ylabel('P of Direct Run')
title('Probability DTSP<1.5')
legend('Session 1','Session 2','Session 3')
hold on   
     


 

 
 

%     

 