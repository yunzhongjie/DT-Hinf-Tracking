%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% H_inf tracking control for linear discrete-time systems          %%
% Model-based PI algorithm                                         %%
% By Yunjie Yang. 2020.                                            %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clc, clear;
warning off;

% F16 Aircraft Systems: Continuous-time
Ac = [-1.01887,   0.90506,  -0.00215;
      0.82225,   -1.07741,  -0.17555;
      0,          0,             -1];
Bc = [0, 1;
      0, 0;
      5, 0];
Cc = [1, 0, 0];
Dc = 0;

%Discretization using the zero order hold (ZOH) method 
sys = ss(Ac,Bc,Cc,Dc);
sysd = c2d(sys,0.1,'zoh');
A = sysd.A;
B = sysd.B(:,1);
E = sysd.B(:,2);
C = sysd.C;

%Reference trajectory system
F = [1];

%Parameters setting
Q = 100; 
R = 0.01;
alpha = 0.1;
gamma = 0.2;

% The augmented system
Q1 = [C'*Q*C -C'*Q; -Q*C Q];
G = blkdiag(Q1, R, -gamma^2);
T = blkdiag(A, F);
B1 = [B; 0];
E1 = [E; 0];

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Calculate the kernel matrix P by directly using the tacking GARE
tempA = sqrt(exp(-alpha))*T;
tempB = sqrt(exp(-alpha))*[B1, E1];
tempQ = Q1;
tempR = [R 0; 0 -gamma^2];
P_dare = dare(tempA, tempB, tempQ, tempR);

% Calculate the optimal control policy and worst case disturbance using P
L_P_dare = inv(R + exp(-alpha)*B1'*P_dare*B1 - exp(-alpha)^2*(B1'*P_dare*E1) * inv(exp(-alpha)*E1'*P_dare*E1 - gamma^2) * (E1'*P_dare*B1))...
    * (exp(-alpha)^2*(B1'*P_dare*E1) * inv(exp(-alpha)*E1'*P_dare*E1 - gamma^2) * (E1'*P_dare*T) - exp(-alpha)*B1'*P_dare*T)

K_P_dare = inv(exp(-alpha)*E1'*P_dare*E1 - gamma^2 - exp(-alpha)^2*(E1'*P_dare*B1) * inv(R + exp(-alpha)*B1'*P_dare*B1) * (B1'*P_dare*E1))...
    * (exp(-alpha)^2*(E1'*P_dare*B1) * inv(R + exp(-alpha)*B1'*P_dare*B1) * (B1'*P_dare*T) - exp(-alpha)*E1'*P_dare*T)

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Model-based PI algorithm

% Initial values of K and L
K = [-5 0 0 5];
L = [-5 0 0 5];
% Initial values of states
x = [0 0 0]';
r = [2];
X = [x;r];

zbar(1:10, 1:20) = 0;
d_target(1:20, 1) = 0;
mm = 0;
Count_N = 1000;
Count_train = 800;

for n = 1:Count_N
    
    if n <= Count_train
        a1 = 1;
        a2 = 1;
    else
        a1 = 0;
        a2 = 0;
    end

    uu = L * X(:,n);
    ww = K * X(:,n);
    %case 1
    u = uu + a1 * (sin(2*n) + sin(5*n) + sin(9*n) + sin(11*n));
    w = ww + a2 * (cos(3*n) + cos(6*n) + cos(8*n) + cos(12*n));
    %case 2
%     u = uu + a1 * 10 * (rand-0.5);
%     w = ww + a2 * 10 * (rand-0.5);

    x(:, n+1) = A*x(:,n) + B*u + E*w;
    y(n) = C * x(:,n);
    r(n+1) = F * r(n);
    X = [x; r];
    
    if n==300
        r(n+1) = 4;
    end
    
    if n==600
        r(n+1) = 6;
    end
    
    % Data collection
    zbar(:,1) = zbar(:,2);
    zbar(:,2) = zbar(:,3);
    zbar(:,3) = zbar(:,4);
    zbar(:,4) = zbar(:,5);
    zbar(:,5) = zbar(:,6);
    zbar(:,6) = zbar(:,7);
    zbar(:,7) = zbar(:,8);
    zbar(:,8) = zbar(:,9);
    zbar(:,9) = zbar(:,10);
    zbar(:,10) = zbar(:,11);
    zbar(:,11) = zbar(:,12);
    zbar(:,12) = zbar(:,13);
    zbar(:,13) = zbar(:,14);
    zbar(:,14) = zbar(:,15);
    zbar(:,15) = zbar(:,16);
    zbar(:,16) = zbar(:,17);
    zbar(:,17) = zbar(:,18);
    zbar(:,18) = zbar(:,19);
    zbar(:,19) = zbar(:,20);
    temp0_zbar = [X(1,n)^2;       X(1,n)*X(2,n);   X(1,n)*X(3,n);  
                  X(1,n)*X(4,n);  X(2,n)^2;        X(2,n)*X(3,n);
                  X(2,n)*X(4,n);  X(3,n)^2;        X(3,n)*X(4,n);  X(4,n)^2];
    temp1_zbar = [X(1,n+1)^2;         X(1,n+1)*X(2,n+1);   X(1,n+1)*X(3,n+1);  
                  X(1,n+1)*X(4,n+1);  X(2,n+1)^2;          X(2,n+1)*X(3,n+1);
                  X(2,n+1)*X(4,n+1);  X(3,n+1)^2;          X(3,n+1)*X(4,n+1);  X(4,n+1)^2];
    zbar(:,20) = temp0_zbar - exp(-alpha) * temp1_zbar;
     
    d_target(1,1) = d_target(2,1);
    d_target(2,1) = d_target(3,1);
    d_target(3,1) = d_target(4,1);
    d_target(4,1) = d_target(5,1);
    d_target(5,1) = d_target(6,1);
    d_target(6,1) = d_target(7,1);
    d_target(7,1) = d_target(8,1);
    d_target(8,1) = d_target(9,1);
    d_target(9,1) = d_target(10,1);
    d_target(10,1) = d_target(11,1);
    d_target(11,1) = d_target(12,1);
    d_target(12,1) = d_target(13,1);
    d_target(13,1) = d_target(14,1);
    d_target(14,1) = d_target(15,1);
    d_target(15,1) = d_target(16,1);
    d_target(16,1) = d_target(17,1);
    d_target(17,1) = d_target(18,1);
    d_target(18,1) = d_target(19,1);
    d_target(19,1) = d_target(20,1);
    d_target(20,1) = [X(:,n); u; w]' * G * [X(:,n); u; w];
    
    %The number of independent elements of P is 10.
    if mod(n, 20) == 0
        err_K = abs(K - K_P_dare);
        err_L = abs(L - L_P_dare);
        
        if n <= Count_train
            mm = mm + 1;
            % PEV using the LS
            Z_temp = zbar * zbar';
            T_temp = zbar * d_target;
            rank(Z_temp);
            vecP = inv(Z_temp) * T_temp;
            P = [vecP(1),    vecP(2)/2,   vecP(3)/2,   vecP(4)/2;
                 vecP(2)/2,  vecP(5),     vecP(6)/2,   vecP(7)/2;
                 vecP(3)/2,  vecP(6)/2,   vecP(8),     vecP(9)/2;
                 vecP(4)/2,  vecP(7)/2,   vecP(9)/2,   vecP(10)];
           
            %PIM
            L = inv(R + exp(-alpha)*B1'*P*B1 - exp(-alpha)^2*(B1'*P*E1) * inv(exp(-alpha)*E1'*P*E1 - gamma^2) * (E1'*P*B1))...
                * (exp(-alpha)^2*(B1'*P*E1) * inv(exp(-alpha)*E1'*P*E1 - gamma^2) * (E1'*P*T) - exp(-alpha)*B1'*P*T);
            K = inv(exp(-alpha)*E1'*P*E1 - gamma^2 - exp(-alpha)^2*(E1'*P*B1) * inv(R + exp(-alpha)*B1'*P*B1) * (B1'*P*E1))...
                * (exp(-alpha)^2*(E1'*P*B1) * inv(R + exp(-alpha)*B1'*P*B1) * (B1'*P*T) - exp(-alpha)*E1'*P*T);
           
            err_L_norm(mm) = norm(err_L);
            err_K_norm(mm) = norm(err_K);
        end 
    end    
end

%The learned results
L_PI = L
K_PI = K

%%
%Figures
figure(1), hold on;
subplot(2,1,1), hold on, box on;
plot(1:mm, err_L_norm,'m','linewidth',3);
plot(1:mm, err_L_norm,'bo','linewidth',3);
set(gca,'FontSize',30, 'FontName','Times New Roman');
ylabel('||L-L^*||');
subplot(2,1,2), hold on, box on;
plot(1:mm, err_K_norm,'m','linewidth',3);
plot(1:mm, err_K_norm,'bo','linewidth',3);
ylabel('||K-K^*||');
set(gca,'FontSize',30, 'FontName','Times New Roman');
hold off;

figure(2),hold on, box on;
t=1:Count_N;
plot(t, y, 'm', 'linewidth', 2);
plot(t, r(1:Count_N), 'b', 'linewidth', 3);
set(gca,'FontSize',30, 'FontName','Times New Roman');
legend('Output', 'Reference');
hold off