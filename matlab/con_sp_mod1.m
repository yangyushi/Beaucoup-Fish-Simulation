% MATLAB code for the constant speed model used to simulate the behavior of zebrafish schoals
% confined in a two-dimensional circular tank. The model assumes that the fish move with a 
% constant speed v0 and change their orientation according to overdamped equations of motion, 
% in response to external forces and interactions. The model includes a wall-avoidance force 
% and Gaussian noise. The fish-fish interactions consist of repulsion-attraction and alignment 
% interactions within the field of view. A form of hydrodynamic interactions is also included.
% The outcomes are the x and y coordinates of all the fish denoted with "xol" and "yol" and 
% their orientation angles "phi_ol". All these are arrays with size(Nfores*N, Np) with "Nfores"
% denoting the number of sample iterations, "N" the number of time steps and  "Np" the number
% of particles.
%Author: Alexandra Zampetaki ,2023

% for octave
pkg load mapping;

Nfores=1; %sample iterations
Np=50; % particle number
N=2000; % number of time steps

tsav=1; %save every step
dsav=1/tsav;
Nsav=N*dsav;
dt=0.067; %time step
Dr=0.01; %rotational noise
Dt=0.00; %translational noise
v0=1;% fish speed
R=66.7; %tank radius
at_str=1.0; %existence of attraction interactions (1: existent, 0 non-existent)
al_str=1.0; %existence of alignment interactions (1: existent, 0 non-existent)
hyd_str=1.0; %existence of hydrodynamic interactions (1: existent, 0 non-existent)

ext_str=3; %strength of external/wall force
max_rot1=pi/6; %maximum rotation angle
ap=8.1*10^(-3); %for external/wall force

%attraction  (Morse potential)

xmin=1; %minimum of attraction potential
lrep=0.67; % length-scale of repuslion
latt=5; %length-scale of attaction
catt=3; %attraction strength
crep=catt*exp((1/lrep-1/latt)*xmin); %repulsion strength

%for perception cone
eps=0.3; 

%alignment
lal=1.33;%length-scale  of alignment
cal=0.075; %alignment coefficient

%hydrodynamics
lhyd=1.33; %length-scale of hydrodynamic interactions
chyd=0.25; %coefficient for hydrodynamic interaction


%initialization
xol=zeros((Nsav)*Nfores,Np); % save x coordinates for all samples
yol=zeros((Nsav)*Nfores,Np); %save y coordinates for all samples
phi_ol=zeros((Nsav)*Nfores,Np); % save velocity orientation angle for all samples

for itim=1:Nfores %sample iterations
    
%initialization
phi1=zeros(Nsav,Np); % velocity orientation angle for each sample
x1=zeros(Nsav,Np); %x coordinates  for each sample
y1=zeros(Nsav,Np); %y coordinates for each sample

%random initial orientation and coordinates
phi=2*pi*(rand(1,Np)-0.5); 
x=(R/2)*(rand(1,Np)-0.5);
y=(R/2)*(rand(1,Np)-0.5);



for i=1:N %iteration in time
    
    att_force=zeros(1,Np); %attraction force 
    al_force=zeros(1,Np); %alignment force
    hyd_force=zeros(1,Np); %hydrodynamics force

    r=sqrt(x.^2+y.^2); %distance of fish from center of tank
    f_wall=(exp(-ap.*(r-R).^4)).*(x.*sin(phi)-y.*cos(phi))./r; %wall potential
       
     
    for j=1:Np %iterate in j-th  particle
      sumij=0;
     
        for k=1:Np %iteratein k-th particle
            if (k~=j)
                %calculation of interactions
                xjk=(x(j)-x(k));
                yjk=(y(j)-y(k));
                
                dphi=phi(k)-phi(j); %orientation angle difference between fish j and k
                dist=sqrt(xjk.^2+yjk.^2); %distance between fish j and k
                
                cvis_an=-(xjk*cos(phi(j))+yjk*sin(phi(j)))/dist;  %cosine of perception angle
                svis_an=-(-xjk*sin(phi(j))+yjk*cos(phi(j)))/dist; %sine of perception angle
                
                visij=(1+eps*cvis_an);  %accounting for perception cone 
                
                thkj=atan2(-yjk,-xjk); %perception angle
                
               
                    %attraction force calculation on fish j
                    att_force(j)=att_force(j)+(crep.*exp(-(dist/lrep))-catt.*exp(-(dist/latt))).*sin(phi(j)-thkj).*visij;
                    %alignment force calculation on fish j
                    al_force(j)=al_force(j)+cal.*exp(-(dist./lal)).*sin(dphi).*visij;
              
                    %hydrodynamic force on fish j
                    hyd_force(j)=hyd_force(j)+chyd.*(1./(dist.^2)).*sin(2*thkj-phi(j)-phi(k));
         
             end
        end

    end
%toc

%calculation of orientation angle change
ddphi1=sqrt(2*Dr*dt).*randn(1,Np)+ext_str.*f_wall*dt+at_str*att_force*dt+al_str*al_force*dt+dt*hyd_str*hyd_force;
ddphi=wrapToPi(ddphi1);  %wrap change of angle in interval [-pi,pi]
    %update angles and coordinates
    phi=phi+min(abs(ddphi),max_rot1*ones(size(ddphi,1),size(ddphi,2))).*sign(ddphi);%maximum rotation
    phi=mod(phi,2*pi); %wrap angle in interval [0,2*pi]
    x=x+dt*v0.*cos(phi)+sqrt(2.0*Dt*dt).*randn(1,Np);
    y=y+dt*v0.*sin(phi)+sqrt(2.0*Dt*dt).*randn(1,Np);
    %save updated coordinates
    if(mod(N,tsav)==0)
    c=fix(i/tsav);
    x1(c,:)=x;
    y1(c,:)=y;
    phi1(c,:)=phi;
    end
end

%save for  all samples
xol(1+(itim-1)*(Nsav):(itim)*(Nsav),:)=x1;
yol(1+(itim-1)*(Nsav):(itim)*(Nsav),:)=y1;
phi_ol(1+(itim-1)*(Nsav):(itim)*(Nsav),:)=phi1;

end

%%For the case of three fish, else comment out
%positions of 3 fish a,b and c
%ra=[xol(:,1),yol(:,1),zeros(size(xol,1),1)];
%rb=[xol(:,2),yol(:,2),zeros(size(xol,1),1)];
%rc=[xol(:,3),yol(:,3),zeros(size(xol,1),1)];

%velocities of 3 fish a,b and c
%va=v0*[cos(phi_ol(:,1)),sin(phi_ol(:,1)),zeros(size(xol,1),1)];
%vb=v0*[cos(phi_ol(:,2)),sin(phi_ol(:,2)),zeros(size(xol,1),1)];
%vc=v0*[cos(phi_ol(:,3)),sin(phi_ol(:,3)),zeros(size(xol,1),1)];

save -mat result.mat xol yol phi_ol
