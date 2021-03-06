install.packages("glmnet")
install.packages("flare")
install.packages("gam")
install.packages("randomForest")
install.packages("foreach")
install.packages("doParallel")

library(devtools)
install_github("vqv/ggbiplot")

setwd("E:/master/OnlineNewsPopularity")
#setwd("/Volumes/CLASS/master/OnlineNewsPopularity")
data <- read.csv("OnlineNewsPopularity.csv") #url, timedelta are non-predictive
summary(data) #something data point are wrong

sum(data$n_unique_tokens > 1)
which.max(data$n_unique_tokens) # data$n_unique_tokens[31038] = 701, we can't get a rate like that. So as this point other two rates, remove!
data <- data[-31038, ]

#check data channel
#channelthing <- data[,14:19]
#chansum <- rowSums(channelthing)
#sum(chansum == 1)
#sum(chansum == 0)
#sum(chansum > 1)  # no article in mutiple channel, some are in no channel

#channelthing$data_channel_is_none = chansum == 0
#channel <- factor(apply(channelthing, 1, function(x) which(x == 1)), 
#                  labels = colnames(channelthing)) 

# check the weekday
#weekdaything <- data[,32:38]
#weekday <- factor(apply(weekdaything, 1, function(x) which(x == 1)), 
#                  labels = colnames(weekdaything)) 

#data <- data[,-c(1, 14:19, 32:38)]
#data$weekday <- weekday
#data$channel <- channel

for(i in 2:61)
{
  f_name <- names(data)[i]
  filename <- paste0('feature/', f_name, '.png')
  png(file = filename, width = 480, height = 480, units = "px")
  hist(data[,i], xlab = f_name, main = f_name)
  dev.off()
}

no_norm <- data[, c(2, 14:19, 32:39)]
#no_norm[, -1] <- lapply(no_norm[, -1], factor)
be_norm <- data[, -c(1, 2, 14:19, 32:39)]
mean_norm <- colMeans(be_norm)
sd_norm <- sqrt(colSums((t(t(be_norm) - mean_norm))^2)/39642)
after_norm <- t((t(be_norm)-mean_norm)/sd_norm)

data1 <- cbind.data.frame(no_norm, after_norm)
data1_l <- data1
data1_l$shares <- log(data$shares)
share_l_mean <- mean(data1_l$shares)
share_l_sd <- sqrt(var(data1_l$shares))
data1_l$shares <- (data1_l$shares - share_l_mean)/share_l_sd

# Something from orginal paper
# e1071 for svm
# randomForest for rf

D1 <- 1400
popular <- as.numeric(data$shares >= D1)
png(file = 'feature/popular.png', width = 480, height = 480, units = "px")
hist(data$popular, xlab = 'popular', main = 'popular')
dev.off()
train <- data[1:9000,c(3:60,62)]
test <- data[9001:10000,c(3:60,62)]

library(e1071)
m1l <- svm(formula = popular ~ ., data = train, kernel = 'radial', type = 'C-classification')
m2l <- svm(formula = popular ~ ., data = train, kernel = 'linear', type = 'C-classification', cost = 2)

pred1 <- predict(m1l, test)
mean((as.integer(pred1)-1-as.integer(test[,59]))^2)

pred2 <- predict(m2l, test)
mean((as.integer(pred2)-1-as.integer(test[,59]))^2)

m3l <- svm(formula = popular ~ ., data = train, kernel = 'linear', type = 'C-classification', cost = 2^2)
pred3 <- predict(m3l, test)
mean((as.integer(pred3)-1-as.integer(test[,59]))^2)

m4l <- svm(formula = popular ~ ., data = train, kernel = 'linear', type = 'C-classification', cost = 2^3)
pred4 <- predict(m4l, test)
mean((as.integer(pred4)-1-as.integer(test[,59]))^2)

svm_tune <- tune.svm(popular ~ ., data = train, kernel = "radial", cost=2^(0:6))


# My work
# linear
library(faraway)

linearm <- lm(shares ~ ., data = data1)
step(linearm)
step(linearm, k = log(39643))

linearm_l <- lm(shares ~ ., data = data1_l)
step(linearm_l)
step(linearm_l, k = log(39643))

# after BIC step, it goes to lm(formula = shares ~ n_tokens_title + num_hrefs + average_token_length + kw_min_avg + kw_max_avg + kw_avg_avg + self_reference_min_shares + global_subjectivity + avg_negative_polarity + timedelta + data_channel_is_entertainment, data = data1)

data2 <- data1[, c(1, 3, 16, 21, 25, 33:36, 44, 53, 60)]

cv_linear <- function(data, form, cv_num, seed = 250)
{
  set.seed(seed)
  n <- dim(data)[1]
  cv_cut <- sample(1:n)
  errors <- rep(0, cv_num)
  for(i in 1:cv_num)
  {
    test <- data[cv_cut[round((i - 1)/cv_num*n + 1):round(i/cv_num*n)],]
    train <- data[-cv_cun_	` ��2������NT<�K�&��o}���0��k�u�׭�e��P�p����u�'��ԲF>��N�=YSs�pw������͕#������l6���������M�F� *o%���=_��#���Ʋ'�n��ӎ��T,*t�����N�cQݢMt�Q��k��boPd�')\��:}e�weiɏ��1��~�����'�Ƶ�P㡧�T�G�{���ס��@�Lm����؎؏���.�4��S�NR�8F��Sͱ9�J�t�I��͈�!i��y���[>�wka%1L���N׉��B87���X� !ܙZ�UʔE��դ��d$�D���j�'MMΒ�=����TH��z�_U�}~��ʩ4Ξ�� �Gc����vR����QF��d�������yv�M���H��9�V�x�x��*��t�T����ۼ��{��Y�Ds�� �m.�E��d����}=΂���^\~C�B>?f3ǼL+��h I�^�֗~S�Y ��T[�(���TL����(�����rC��F>j^����絤�i�f�mzW�����[��Z)�5O�v���h�%�f�a��6�ޚ#x�5����ǯmA����uQO�kX����Z0>� )��d9`�[*F�U"sK7+�%�M3��%��c���z}�v.B���I�Y�;/�=��[�t� ����Q�g3��%J�w�=���e(�|&T�6I9��)� ����X��@Td5�$	J*�i��|z=�_���	�Nە'[/ݡ��	��I�;K��*����m��X��X��Xͺ�����Q�Ft�]>� �J�vl�,��GbQtis�|3��g�G�cWc�(�*�*��-c'����*����*DGKU\$?��>jז�d�O�Z���p)�)mZ�;�'B��
�e2CO������s�c��7�$��,C�H�O
�h�13�q�s��d6��0��xf~�tU��ʲ�g����KE6~�Q�,������%���w�E�IZ�<lEId����Maڶ�鑵
GlD}f�'�c��l�F���`��ת'%��r�0����MM��kCe�\�����o8��v�-$F����'��b9��X�(n��	�׳��i{�$UF:��֫��zӾ�E���R�Wy���"�F�ϜŗK�?qH׺�}:{i���Mյ若�w.7A�C�sc�	��"������(AWf�NM�䏒{�	jp���u�vK��ȼ?X�c��Sg��ol�D�#D-�L�2��Ol�����>%��_�;(��ٶ��t��2���*F�I�'�~��*J�����S�)�rJ�9�2�*�a����g�<^Mb�ؗ9���'�(&!�7	��\l�c��y_p ��"%��~�n}��`6�ϕE*H��QImDE�>	�A�~_�F� �i�Cp�[�nFe:��
o�M��}����i��;c���bj��(�L������Xua��~��{�,儤�Lg����7J��S	X�
��c��uc� ��~9�v�/�<�vUe��Y���y#���eד1��&EL�Yat˜5������]1+�g�&��JRU	
.ݱA*1�/AIl�)����$�V��Z����{�dk89���6?/�8	�ړ���V�Rᤜ��;}�jS�P�>��a6��Ԫ�^V��u��A	��6���y'k{|����&��#�޽zku�hb��kL��ÏC�Ի\�P�����j��s��*�mV�;��]o%��$]�Up��P�z�T���o��׿�j�ٗ�R�I+B�Ϟ۝ɜE�>M~36�r�ؓ�Nd�ؿ�aN�5�7�]�gg�䱪F�t�E֔�p��Pt7�T����r}�2��\�񘜶���Iwv����LY���mW*I�酁�J�i��tO1Jq���g�&9�d��9�3�����7lC�gGx���Um�w���������tp������A��C)JK��T�q}$'��pO*ϡZC�>!-on����-R��Ău��=�M��p=t���'��=)�x6*>xC��9��a�Je<v :n�v���=���k�(��H4�R��7�U��@�]S�˳Bʯ����S~iQ��r_t�������h����6S�Mǘ��I9�"�8%.��$5�A]��IG�a&�px	��n�c��bNh:f<���ԌD�-H<|)'H���L����i�"r7/����20-��q�o�.��{i�	|s������:�A8M��ȣ*'�|��~3�/d�E-�s���aT�EMCr��|���M�p��8������Q|m�)�O0ys������p1Xr���ޝYg?�ւ���=$�D3������/��ǥE��wD����C��ѻi���%"Ж,L��[2�co����7�hY����$d�rT�}�'7L����ش0ؼ]x�e�����F=c$!m�^��y��Z�.2]�b���wS��u��}����D��h]�g���Y�߿��ﯧV̔]�����5`P��ʇ*ƌ�-�����/M�'�퐳�W�rw�kd���r��y�����O�q̯��~쫏O]$�Q���N�#6>rC�/�E06r����v������>K������M{z�%R�ϟJ�z�;�un�`��g0�n;.Ͱ���%5�Ef-��2��Z�C�D ��V��l�e݋�!�h������
endstream
endobj
530 0 obj
<<
/Length1 1632
/Length2 8015
/Length3 0
/Length 9071      
/Filter /FlateDecode
>>
stream
xڍuT�k�.!���C7��� R*9�0�C� �J)���H���%�����}�����9k֚y����z��~��Yߐ_����!h~� P
��ch�! ��@!Bvv#8�	�;!�	�G"��`(�``�M��!�  Mw' H ��K� ! P�?D$J
���C: M$�FȮ�t�F����7u����p@���|����0F t�h{��ME�	`���ah�����G�]�===��nH��7���<���P0(��� ]�3����F�p���H[�'�����&��� 7��� =�/��_>�߇ 	������_�����`��Fx�v [����-��B���/"��y� Ý�67�߭�j� �̈́��A�]�nnp�_3
�Jss̪�2���@���O��An��[���uD =��A�p���PwAc�����7��D������@	� �
�yA�0�v��v�~�of��uA� loƀ��ma7?��n` �r������7"� P8������d�1�l��7���{̀7���>�}��Q�p�������u�*���{��:���^ _~ЍF��D� HX .��;�>�w#��5�H��_���z��[\o7�߹t�7҅��Q�9P�����w��M濲�?���;Rswr���������p'�7�uG߬��f��j
�kwu`P�����j��7레���4?H�/#�M�������4���׶9�0}����������Y1���;��F��]���wQU	��jB�b 0
�&���$
���$��[� A}��`�D��UqQ���/�o$!4�/�A`����@@�������� A�?�@���Ɍ�
]�En2�ܨ�Gm�M1�?��FA7����� A���M����b>���Bݼc~��������`0/�pz	�q�	i>�R���_����v�P���ݭ��j9�0;kF��tH�ҡMW��,ga��w�S���	?�ږ�M���	�X���=�	�F�|�%F�bT7�>C4y;�<Ԯې�̏���T�'ǝ5q��f�����,m�{�&�+�om���E�i8sW��Wc�#Su3X�dZ�	<+ɗz��nw1tDiz<]�gC�V�"�Wi2���Gn��N���}Je*C�|��t剈�>BϹ�gx���"Á��| �ۇ������T8i�΋��k�wާ8͞�Ed�_��PV�03(�[>n�'f��K�j�n��=��_ص�Xm�_���}E�Z.I�ܮ�h!Wv!���e{ˠm<�,�����Qcc�4ʆb&��~��s? L�'(�rZUΆq7bK�-e�hx�Zr�oq�K����)�D��t6f�g$�x�T;�P�>�[ъ�-?ŗʱ�}��s.2�JG-��
�+ר�7|��~�����9��Z�7w�[��L��I<��|��u~857r�0��ָ�P���3��#&c��V���*�⼙.��+�P��~�5���0{�'�(�'��3c� �x�&[��[D[���VJ�����X9��1��b�}�;%�Q�z�)���+I��G#(�4�mZ"��i���O �Sʵ��f^J�����o��i��s(ğl�V��)��(�ݕ�>�7���b~�b�1P�T�g���>k���W	�C�Q��K�Y���ʆ���^�T�*��o�21�j����G�YpK��aQE�"�Ǉ|Qc�����=�w���2#��,�t�;��L��No�j!��ª�:׻�*
�~Q��u�Q	��`���M�����Q����g���~(|RE�����y����t�ٜ�����&��m=u0H�����kC�"���G�<���vp%Yp�>���B�@��s��(:PJg\3P�7`o���1v���E��|����|h�NҠa�sȑ޵�{E5�9񽟨(��-,jEg�5�}y@si9�G�wvϣjO�TobMy�0)�S��:i��l�U|���>'����	��գO��/�uuCWwn�x'}�.��]Q�
� ��w���0�o=�Y��;�fj\�5���b�fVE�J�Oe���|��]1R��2.H��CV�!�U�����q`7�"l���d�__~�G"�{*�L�0��|S�̄�C:x�RMYߨ@�v;�;0w\ů���wB4	6�T����8u��h��yίh�����ز�nG���@K(vV2?�.wgq���ݟM,��Q,5�����A�ǈQE�.�(C�;!�M\��\�3���<��K�؅��<D�Tt[��PXKH�؀+f���܋y5j�j6U�ob���r���(]y�^��*4���7o�?"�B�ډqSg�^�n�
�7��\�Py'�ZΧ�9&F���û��F������3��zYd�S[m��&���a�T����\7	O��K_i���+�Fe�F_���{��Ǯ�EϢ�o�6	=��Kb�o%�b�az+�EN�ͷ3I�d�/����q�D��$�0�5y���l�O����j��g�q�±c'��kLv���<�伇(_�Eyᨥ���v�PF�������+�U���EXڳ����C\�՛s�g�Fvצ^��|]�N'E����Մ:<��l��/�U�<��g�>(lՙl5Мԕwq�4[�S�]6��؉X��Q[��+�|��5_�����a1��o- @�Dm�<>i��
��D��S@cD�08H�i����H7�/�$�YQ���H�<���%5p����<�_J�PŅ:� G��$���[7Z*���#b������p��gMH��q�m�� w����/#�>�_�9d6,�ϻ���=7�8i幒ƽ�����nPsɌ�U�J�/�?�����ڿq`~�n���4@K���7�\Y��΀�D̖Ŵ���,�X�!�/��-�}�B�1��,�%^L�f���}�P�=��`U�7�l�ކ��f�&����$��EW���ݡ���xS'���}�,
"<d���\
��[���',C�?�Vl_��W���d�,��xO�[Q�w�Lg5��f�Fx9�\����H1��'���e�V?��.������~"�y2Mz�gA��G������(i�%ȇw�;��r~�5�ngbJ�l�AG&2�p����Zd��_JI�x�#���}g¬G�AғT��A���g�|�v�����(�8��R���aJR��܋w���$K�:����s?��w/ߴF�(Eb��f~'`p���q_qgw�$+�`��'�^�jMA5�����I�4_G״߲7妊t��kG'����D��~���;6{�z~���#�z���}čg���2�� ���7�v���G�wh����J+S%����X�ZS7��r��c^�Jp���S� Yǁ���-�*��S�g�������,j�ߘN,\.V;x�N�� (��8��X�j���C;.gN�_�9oÐ�㘐�r[�@Jā�����`Ʃ�=������`�e�C��sU,A����I��|��K���:�"aUk}:M��5�ԅ���o2�����I���%;��{
�u�|��\��v�ق\��	]+X謴��&��_��<�V��r�'�����~l[��& {��D�P������#N�]�̦��.�(���BO�	��iqbRh21R)o3�%y�O�iϼU��6�?G/�A���Y�����j��qf��(�r��)�Ľ�k�Q���(K�a��)U	��Dg�>9�J��7�d�q�6�㴭��s'tD����(T�̋�
�U�t��`#)��%in�)�b���c��4�fxS�`��Ӫ��ɾK�ʬ9ܱ��[�ѧ�Q!�zߢ�A�cZ{D$H��C�ɠd�/��)���ݰ�*�=Np\�B��y1~f�BW�5u9R#��cv]j,�́����bFZ��ۙ�H���0��hQ��TYM��9|����(%})VO�.��u�'8
Z�MBZ2��8���a�c���Y؀n�ޞ#	n3&6g&u){E�<=s�J��5��}�׀�fV�UܖR!�?�*w��-?��[�:Y������0۷$�J��I��1>��u��>��,�Fզj�/N�G��^If�/-�є��3��~z�^̼�<[�Q��pJ�q��Lӊ O:�����;�U��,�e9�^^�KDd��?0ɍ��q9��{7}����ML�X�~e�?��xkN.)��z/b��hP)BP'(�ҳDN7k��a�~��`���r�ŝ���Ѷ�OpdI�X����ΞVa�'��
�uF�b�F�.oTf�4�]*�M�U�Š�+V����$�}�\��WQ�^cg뭍��<���}�b���w5��?ޱ�I�������0�t��L���=�i"ӡ��=3���N�;P�k77�Q��-u�}�n�4+���ԡ�w ��u*�cMPr��J�����nt����M�K�:���>�s����=��J�d�$��ع"�Yʇ�������%�%	<���*�C>��I<�6r�(�*��:-�&�Q�Q�(���|y��\��Sa`+�n;�f��\ �1���)v�Y�w4(��8&٫�>�n؇�W7�%��%z���3�Lh?B7���T�dH���ǈ�3*�г���Y)��D��>4S��ƢJ�Gw�3h�⿿_&���%����9 ����Si���q8	=	��G���/I>��9���'�� 5��!��ev�j�%���1*ܻ�O��+�1�öj]�Xk q�V�G�ZVt���UC]y�!���#�1��"@��O��+�I�bm��.�^H��j Ul4/(ƥ�SY�^� ���<CJ� ث&�CmL>1T�X[�AՈf�[�5Q��>0w�<����2��u<{n��EtV)߻�!R��}�|�1r�b؟Ү����Ldv���z���hRs���ǣ�m�A6�/
!��][XB�c_�B5m[��IZ[��b�ך��[�}+e>��,�z��|��Sg����Pk�K���㼱ӓWg���sK>J5F'Ѭ�ߢ���;N�NL�O���A.��>���o�
.Җ�S��� 3Km�n�w/2�{�N#�xӰ@��2L�ٯ=O�\�P5V�8JUx��kj�K�%�V�ܛ��=	&�lP�Sq���8w ��J���}N�`��:��n����}+2��tS��BS��4ց�Ҷb������e-WP�W�o����v���ئ��:�cq�Խ"1�o_ы��W��(���:>��{�6�L�_}��T������cl۲"�[��u��z�^�Ζ��&����Y:�c.۴>�XKd�1߰�c�y؏3��m���y�	ұy�^|�5�*Q�NA��Ae����5��o�K�5tH���:�|��&�g��nVE��|�N��V�3>�"�"�W��;*��~R���fJ��m*EU�9��5���Q٭��un�{":k�Ӓ�
�I��=𸵅�ѫ-�/j�6ZT*�ZK�c:�_�s��ߡЪ�'o�ӯf�L�\�s<o����?�x5��!�R�����N����u<���^��;�Y�fj���y6�{Ƶ� ^����v��wV���_���g�Nc��(��e�"�)ݭdh�z�B� yiI\�ÛD