%% Code Providing a Demonstration of the behavioral analyis presented in the paper 
% "Learning-induced shifts in mice navigational strategies are unveiled by a minimal behavioral model of spatial exploration"
% Here distributions of Trial Lenghts obtained from simulating mice behaviour are compared to the one coming from experimental data.  
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
goalrun_plist = [0.4 1 1.2 1.4];  % Foresight parameter (adjust for each trial group)
longrun_p = 0; % Probability of starting a long run

n_rep =10; % Number of simulated runs for each experimental trial

%%



% Cycle through trial groups
for tri=1:1
goalrun_p = goalrun_plist(tri);
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

OP_Ratio_Sim = []; %Path Ratio for model trials

tr_r = 0;  % Counter for model trials


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

% And the relative Distance Matrix for all node pairs

Distance_Matrix=zeros(size(Transition_Map_Barrier,1));

for ss=1:size(Transition_Map_Barrier,1)
    for ee=1:size(Transition_Map_Barrier,1)
    [~,Distance_Matrix(ss,ee)]=shortestpath(G_barrier,ss,ee);
    
    
    
    end
    
    
end   


[nodes_path,short]=shortestpath(G_barrier,St_Node,En_Node); %finds nodes path/shortest length

%vector with all the shortest lengths
Trial_Len(tr)=short;








 for rep=1:n_rep
 

 tr_r=tr_r+1;
 Goal=0;
 Length=0;
 

 

 
 In_Diag=0;
 
 Old_Dist=short;
 clear Choice
 while(Goal==0)  
     if(Length==0)  %if he hasn't moved yet,
     C_Node=St_Node;  %current node = starting node
     Last=St_Node;    %and last visited node is the starting node
     end
     

     %Is the animal already on a digaonal run?     
     if(In_Diag==1)    
         
         
         Next=diag_nodes(dn+1);
         dn=dn+1;
         if(Next==Target)
         In_Diag=0;
         dn=0;
         end
     
     %Or is the animal on a random walk?    
     elseif(In_Diag==0)
         
         %Check if switching to a diagonal run 
         go_diag=(rand(1)<longrun_p);
         
         if go_diag
         
         %Establish Target and trajectory     
         Pot_Targ=intersect(find(Border_Map==2),find(Distance_Matrix(C_Node,:)>3));  % Find Nodes on the external ring more distant than 3 steps 
         Target=Pot_Targ(randi(numel(Pot_Targ))); % Select among available nodes    
         diag_nodes=shortestpath(G_barrier,C_Node,Target); % Path of the diagonal run   
                  
        % Start the diagonal run
        
        Next=diag_nodes(2);
          dn=2;
         In_Diag=1;

         end
         
         
         %Keep random walking
         if ~go_diag
                 
    
                 ch=randi(3);
                 Next=Transition_Map(C_Node,ch);  %random adjacent node
     
                 while(Next==0 || Next==Last)   %if this random number is 0 or the last visited node
                 ch=randi(3);                %find a new one next node
                 Next=Transition_Map(C_Node,ch);
                 end
                 %disp([Next rep tr]) 
            
                
        end
     end
     
     
     
     

     
     Last=C_Node;
     C_Node=Next;
     
     Length=Length+1;
     
     
     [~,New_Dist]=shortestpath(G_barrier,C_Node,En_Node);

     % Check wether to start a direct run
     goal_detect = exprnd(goalrun_p);
     
     if(Distance_Matrix(C_Node,En_Node)<=goal_detect) 
         Goal=1;
     end
     Old_Dist=New_Dist;
 end
     
 

 
 OP_Ratio_Sim(tr_r)=(Length+Distance_Matrix(C_Node,En_Node))/short;

 

 end



    
    
    
    

 end

 % Compute Distribution Similarity Statistics 
 
  [h,p,k]=kstest2(OP_Ratio_Sim,OP_Ratio_mice); % Kolmogorov-Smirnov
 
 P_Vec=linspace(1,10,16);
 
 

 
 P1=histcounts(OP_Ratio_Sim,'Binedges',P_Vec,'Normalization','probability')+0.001;
 P2=histcounts(OP_Ratio_mice,'Binedges',P_Vec,'Normalization','probability')+0.001;
 
 P1=P1/sum(P1);
 P2=P2/sum(P2);
 
 KL=kldiv(P_Vec(2:end),P2,P1); % Kullback-Leibler Divergence 
 
 
 % Plot Cumulative Distributions 
 
figure(1)
subplot(2,2,tri)
histogram(OP_Ratio_Sim,'Binedges',[0:0.5:20],'FaceAlpha',0.3,'FaceColor',[1 0 0],'Normalization','cdf')
hold on
histogram(OP_Ratio_mice,'Binedges',[0:0.5:20],'FaceAlpha',0.3,'FaceColor',[0 0 1],'Normalization','cdf')
legend('Model','Mice')
 


end


   
 

%%     


 
 