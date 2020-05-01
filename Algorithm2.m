%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% H_inf tracking control for linear discrete-time systems          %%
% Model-free Q-learning algorithm                                  %%
% By Yunjie Yang. 2020.                                            %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all;
clc, clear;

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

%Calculate kernel matrix H using P
H_dare = [Q1+exp(-alpha)*T'*P_dare*T,    exp(-alpha)*T'*P_dare*B1,      exp(-alpha)*T'*P_dare*E1;
          exp(-alpha)*B1'*P_dare*T,      R+exp(-alpha)*B1'*P_dare*B1,   exp(-alpha)*B1'*P_dare*E1;
          exp(-alpha)*E1'*P_dare*T,      exp(-alpha)*E1'*P_dare*B1,     exp(-alpha)*E1'*P_dare*E1-gamma^2];
Hxx = H_dare(1:4, 1:4);
Hxu = H_dare(1:4, 5);
Hxw = H_dare(1:4, 6);
Hux = H_dare(5, 1:4);
Huu = H_dare(5, 5);
Huw = H_dare(5, 6);
Hwx = H_dare(6, 1:4);
Hwu = H_dare(6, 5);
Hww = H_dare(6, 6);

% Calculate the optimal control policy and worst case disturbance using H
L_H_dare = inv(Huu - Huw*inv(Hww)*Hwu) * (Huw*inv(Hww)*Hwx - Hux)
K_H_dare = inv(Hww - Hwu*inv(Huu)*Huw) * (Hwu*inv(Huu)*Hux - Hwx)

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Model-free Q-learning algorithm

% Initial value of H
H = eye(6);
Hxx = H(1:4, 1:4);
Hxu = H(1:4, 5);
Hxw = H(1:4, 6);
Hux = H(5, 1:4);
Huu = H(5, 5);
Huw = H(5, 6);
Hwx = H(6, 1:4);
Hwu = H(6, 5);
Hww = H(6, 6);
% Initial values of K and L
K = [-5 0 0 5];
L = [-5 0 0 5];
% Initial values of states
x = [0 0 0]';
r = [2];
X = [x;r];

zbar(1:21, 1:22) = 0;
d_target(1:22, 1) = 0;
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
    zbar(:,20) = zbar(:,21);
    zbar(:,21) = zbar(:,22);
    
    temp0_zbar = [X(1,n)^2;       X(1,n)*X(2,n);   X(1,n)*X(3,n);  X(1,n)*X(4,n);
                  X(1,n)*u;       X(1,n)*w;        X(2,n)^2;       X(2,n)*X(3,n);
                  X(2,n)*X(4,n);  X(2,n)*u;        X(2,n)*w;       X(3,n)^2;
                  X(3,n)*X(4,n);  X(3,n)*u;        X(3,n)*w;       X(4,n)^2;
                  X(4,n)*u;       X(4,n)*w;        u^2;            u*w;            w^2];
              
    zbar(:,22) = temp0_zbar;
         
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
    d_target(20,1) = d_target(21,1);
    d_target(21,1) = d_target(22,1);
    d_target(22,1) = [X(:,n); u; w]' * G * [X(:,n); u; w]...
                   + exp(-alpha) * [X(:,n+1); L*X(:,n+1); K*X(:,n+1)]' * H * [X(:,n+1); L*X(:,n+1); K*X(:,n+1)];
               
    %The number of independent elements of H is 21.
    if mod(n, 22) == 0
        err_K = abs(K - K_P_dare);
        err_L = abs(L - L_P_dare);
        
        if n <= Count_train
            mm = mm + 1;
            % PEV using the LS
            Z_temp = zbar * zbar';
            T_temp = zbar * d_target;
            rank(Z_temp);
            vecH = inv(Z_temp) * T_temp;
            H = [vecH(1),   vecH(2)/2,   vecH(3)/2,   vecH(4)/2,  vecH(5)/2,   vecH(6)/2;
                 vecH(2)/2, vecH(7),     vecH(8)/2,   vecH(9)/2,  vecH(10)/2,  vecH(11)/2;
                 vecH(3)/2, vecH(8)/2,   vecH(12),    vecH(13)/2, vecH(14)/2,  vecH(15)/2;
                 vecH(4)/2, vecH(9)/2,   vecH(13)/2,  vecH(16),   vecH(17)/2,  vecH(18)/2;
                 vecH(5)/2, vecH(10)/2,  vecH(14)/2,  vecH(17)/2, vecH(19),    vecH(20)/2;
                 vecH(6)/2, vecH(11)/2,  vecH(15)/2,  vecH(18)/2, vecH(20)/2,  vecH(21)];
            %PIM
            Hxx = H(1:4, 1:4);
            Hxu = H(1:4, 5);
            Hxw = H(1:4, 6);
            Hux = H(5, 1:4);
            Huu = H(5, 5);
            Huw = H(5, 6);
            Hwx = H(6, 1:4);
            Hwu = H(6, 5);
            Hww = H(6, 6);
            L = inv(Huu - Huw*inv(Hww)*Hwu) * (Huw*inv(Hww)*Hwx - Hux);
            K = inv(Hww - Hwu*inv(Huu)*Huw) * (Hwu*inv(Huu)*Hux - Hwx);
            
            %Error calculation
            err_L_norm(mm) = norm(err_L);
            err_K_norm(mm) = norm(err_K);
 
        end 
    end
end

%The learned results
L_mf = L
K_mf = K

figure(1)
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
hold off

figure(2),hold on, box on
t=1:Count_N;
plot(t, y, 'm', 'linewidth', 2)
plot(t, r(1:Count_N), 'b', 'linewidth', 3)
set(gca,'FontSize',30, 'FontName','Times New Roman');
legend('Output', 'Reference')
hold off