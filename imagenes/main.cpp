#include<opencv2/opencv.hpp> 
//#include<String>
using namespace std ;
using namespace cv ;
//Carlos Jesus Fernandez Basso

//Funciones que ayudan.
void pintarVentana(string name,Mat im){
	namedWindow(name,1); 
	imshow(name,im); 
	waitKey();
	destroyWindow(name);
}
void pintar2Ventanas(Mat im,Mat im2){
	namedWindow("ventana1",1); 
	imshow("ventana1",im); 
	namedWindow("ventana2",1); 
	imshow("ventana2",im2); 
	waitKey();
	destroyWindow("ventana1");
	destroyWindow("ventana2");
}
//Pintar lineas
bool valido(Point2f pixel, Mat im){
	if(pixel.x<0 || pixel.y<0 || pixel.x>=im.cols-1 || pixel.y>=im.rows-1){
		return false;
	}else{
		return true;
	}
}
void lineaBN(Point2f punto,int t,int lon,Mat im){
	vector<Point2f> a;
	Point2f p;
	if(t==0){
		
		for(int i=0;i<lon;i++){
			p.x=punto.x+i;
			p.y=punto.y;
			a.push_back(p);
		}
	}
	else{
		
		for(int i=0;i<lon;i++){
			p.x=punto.x;
			p.y=punto.y+i;
			a.push_back(p);
		}
	}
	
	while(!a.empty()){
		p=a.back();
		a.pop_back();
		im.at<char>(p)=100;
	
	}
}
Mat validaroi(Mat im_original, Point2f esquina_roi, int tam_roi){
	Point2f aux;

	aux.x=esquina_roi.x+tam_roi;
	aux.y=esquina_roi.y+tam_roi;

	if(valido(esquina_roi, im_original) && valido(aux, im_original)){
		Mat roi1= Mat(im_original, Rect(esquina_roi.x, esquina_roi.y, tam_roi, tam_roi));
		return roi1;
	}else{
		if(esquina_roi.x<0)
			esquina_roi.x=0;
		if(esquina_roi.y<0)
			esquina_roi.y=0;
		if(aux.x>=im_original.cols)
			esquina_roi.x=im_original.cols-tam_roi-1;
		if(aux.y>=im_original.rows)
			esquina_roi.y=im_original.rows-tam_roi-1;

		Mat roi1= Mat(im_original, Rect(esquina_roi.x, esquina_roi.y, tam_roi, tam_roi));
		return roi1;
	}
}
void lineaColor(Point2f punto,int t,int lon,Mat im){
	vector<Point2f> a;
	Scalar s;
	Point2f p;
	s.val[0]=40;
	s.val[1]=0;
	s.val[2]=0;
	s.val[3]=0;
	if(t==0){
		
		for(int i=0;i<lon;i++){
			p.x=punto.x+i;
			p.y=punto.y;
			a.push_back(p);
		}
	}
	else{
		
		for(int i=0;i<lon;i++){
			p.x=punto.x;
			p.y=punto.y+i;
			a.push_back(p);
		}
	}
	
	while(!a.empty()){
		p=a.back();
		a.pop_back();
		im.at<Vec3b>(p)[0]=06;
		im.at<Vec3b>(p)[1]=25;
		im.at<Vec3b>(p)[2]=180;

	}
}
// x es la longitud de la cruz
void CruzColor(int longitud, Mat im, vector<Point2f> c, Scalar color){
	Point2f p;
	vector<Point2f> a;

	for(int j=0; j<c.size() ;j++){
		for(int i=-longitud;i<=longitud;i++){
			p.x=c[j].x+i;
			p.y=c[j].y;
			if(valido(p, im))
				a.push_back(p);
		}
		for(int i=-longitud;i<=longitud;i++){
			p.x=c[j].x;
			p.y=c[j].y+i;
			if(valido(p, im))		
				a.push_back(p);
		}
		while(!a.empty()){
			p=a.back();
			a.pop_back();

			if(im.channels()==1){
				im.at<char>(p)=color[2];
			}else{
				im.at<Vec3b>(p)[0]=color[0];
				im.at<Vec3b>(p)[1]=color[1];
				im.at<Vec3b>(p)[2]=color[2];
			}
		}
	}
}
// x es la longitud de la cruz
void CruzBN(int x , Mat im, vector<Point2f> c){
	Point2f p;
	vector<Point2f> a;
	
	for(int j=0; j<c.size() ;j++){
		//if(c[j].x+x<im.cols || c[j].y+x<im.rows){
		for(int i=-x;i<=x;i++){
			p.x=c[j].x+i;
			p.y=c[j].y;
			a.push_back(p);
		}
		for(int i=-x;i<=x;i++){
			p.x=c[j].x;
			p.y=c[j].y+i;
			a.push_back(p);
		}
		while(!a.empty()){
			p=a.back();
			a.pop_back();
			im.at<char>(p)=0;
		//}
		}
	}
}
void Calcula_puntos_harris(Mat im, vector<Point2f> &v){
	goodFeaturesToTrack(im, v, 50, 0.3, 10, noArray(), 3,  true, 0.04);//el true indica que usa el detector de puntos harris

}

void pintarpunto(Mat im,Point2f c){
		im.at<char>(c)=255;	
}

vector<Point2f> correspondencia(Mat im1, Mat im2, vector<Point2f> ptos1, vector<Point2f> ptos2){

	Mat matriz_compara(ptos1.size(), ptos2.size(), CV_32FC1);
	Mat tmp(1,1, CV_32FC1);
	vector<Point2f> correspondencias;
	Point2f parejas_indices;

	for(int i=0;i<ptos1.size();i++){
			Mat roi1= validaroi(im1, ptos1[i], 5);	
		for(int j=0;j<ptos2.size();j++){
				Mat roi2= validaroi(im2, ptos2[j], 5);
					matchTemplate(roi1, roi2, tmp, 3);
					matriz_compara.at<float>(i,j)=tmp.at<float>(0,0);
		}
	}

	double max;
	int indice_max, indice_repe;
	bool repetido=false;
	for(int i=0;i<ptos1.size();i++){		//comparo los puntos harris
		max=-1;
		for(int j=0;j<ptos2.size();j++){
			if(matriz_compara.at<float>(i,j) > max){
				max=matriz_compara.at<float>(i,j);
				indice_max=j;
			}
		}
		parejas_indices.x=i;
		parejas_indices.y=indice_max;
		correspondencias.push_back(parejas_indices);
	}

	return correspondencias;
}
//Apartados De La Practica
vector<Point2f> APARTADO_A(Mat im){
	Mat imagen2, imagen3;
	vector<Point2f> vectoraux;
	Scalar c;

	cvtColor(im,imagen2,CV_RGB2GRAY);	
	goodFeaturesToTrack(imagen2, vectoraux, 1000, 0.1, 10);		//calculo esquinas relevantes
	cornerSubPix( imagen2, vectoraux, Size(5, 5), Size(-1, -1),TermCriteria(2, 30, 0.01));   // refino las coordenadas
	c[0]=255;
	c[1]=0;
	c[2]=0;
	imagen3=im.clone();
	CruzColor(5, imagen3, vectoraux, c);
	cout << "\nApartado A:";
	pintarVentana("APARTADO A:", imagen3);

	return vectoraux;
}
vector<Point2f> APARTADO_B(Mat im){
	Mat imagen2, imagen3, imagen4;
	vector<Point2f> puntosaux1, puntosaux2;

	Scalar color;
	//Paso la imagen a blanco y negro 
	cvtColor(im,imagen2,CV_RGB2GRAY);	
	cout<<"\nApartado B:";
	goodFeaturesToTrack(imagen2, puntosaux1, 1000, 0.1, 15);	//calculo las esquinas relevantes de la imagen
	cornerSubPix( imagen2, puntosaux1, Size(5, 5), Size(-1, -1),TermCriteria(2, 30, 0.01));  
	// refino las coordenadas
	color[0]=0;
	color[1]=0;
	color[2]=255;

	imagen3=im.clone();
	CruzColor(10, imagen3, puntosaux1, color);				


	goodFeaturesToTrack(imagen2, puntosaux2, 1000, 0.1, 15, noArray(), 3,  true, 0.04);    
	cornerSubPix( imagen2, puntosaux2, Size(5, 5), Size(-1, -1),TermCriteria(2, 30, 0.01)); 

	imagen4=im.clone();
	color[0]=0;
	color[1]=255;
	CruzColor(10, imagen4, puntosaux2, color);
	//pintar2Ventanas(im4, im3);
	namedWindow("Con detector de puntos harris",1); 
	imshow("Con detector de puntos harris",imagen4); 
	namedWindow("Sin detector de puntos harris",1); 
	imshow("Sin detector de puntos harris",imagen3); 
	waitKey();
	destroyWindow("Sin detector de puntos harris");
	destroyWindow("Con detector de puntos harris");
	return puntosaux2;
}
void APARTADO_C(){

	Mat imagen_compara1= imread("imagenes/basement00.tif",1); 	
	Mat imagen_compara2= imread("imagenes/basement01.tif",1);
	Mat imagen2, imagen3;
	Scalar color;
	Mat imagen4, imagen5;
	vector<Point2f> parejas;
	Point2f punto;
	vector<Point2f> ptos1, ptos2;

	cvtColor(imagen_compara1,imagen2,CV_RGB2GRAY);	
	goodFeaturesToTrack(imagen2, ptos1, 100, 0.0025, 10, noArray(), 3,  true, 0.04);    
	cout<<"\nApartado C:";
	cvtColor(imagen_compara2,imagen3,CV_RGB2GRAY);	
	goodFeaturesToTrack(imagen3, ptos2, 100, 0.0025, 10, noArray(), 3,  true, 0.04);    
	parejas=correspondencia(imagen2, imagen3, ptos1, ptos2);	 
																				
	color[0]=0;
	color[1]=0;
	color[2]=0;

	imagen4=imagen2.clone();
	imagen5=imagen3.clone();

	vector<Point2f> aux;

	for(int i=0;i<parejas.size();i++){
			aux.push_back(ptos1[i]);
			CruzColor(5, imagen4, aux, color);		
			aux.clear();
			aux.push_back(ptos2[parejas[i].y]);
			CruzColor(5, imagen5, aux, color);			
			aux.clear();	
	}
	namedWindow("Comparacion 1",1); 
	imshow("Comparacion 1",imagen4); 
	namedWindow("Comparacion 2",1); 
	imshow("Comparacion 2",imagen5); 
	waitKey();
	destroyWindow("Comparacion 1");
	destroyWindow("Comparacion 2");
}
void APARTADO_D(){
	vector<Mat> im_flujo;
	for(int i=1;i<11;i++){		//leo las 10 imagenes del flujo
		char* aux=new char [100];
		char* tmp=new char [2];
		strcpy(aux,"imagenes/FRAME");
		itoa(i, tmp, 10);
		strcat(aux, tmp);
		strcat(aux, ".bmp");
		im_flujo.push_back(imread(aux,1));
	}

	vector<vector<Point2f>> ptos_encontrados, ptos_harris;
	vector<vector<uchar>> status;
	vector<vector<float>> err;
	vector<Point2f> aux_ptos1, aux_ptos_encontrados;
	vector<uchar> aux_status;
	vector<float> aux_err;
	Mat im_gris1, im_gris2;
	cout << "\nCalculando apartado D:";
	for(int i=0;i<im_flujo.size()-1;i++){	
		cvtColor(im_flujo[i],im_gris1,CV_RGB2GRAY);
		goodFeaturesToTrack(im_gris1, aux_ptos1, 100, 0.2, 5, noArray(), 5,  true, 0.04);	
		ptos_harris.push_back(aux_ptos1);
		calcOpticalFlowPyrLK(im_flujo[i], im_flujo[i+1], aux_ptos1, aux_ptos_encontrados, aux_status, aux_err, Size(15,15), 3, TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), 0, 0.3);//1e-4
		ptos_encontrados.push_back(aux_ptos_encontrados);
		status.push_back(aux_status);
		err.push_back(aux_err);

	}
	
	
	Scalar  c2;
	c2[0]=0;
	c2[1]=255;
	c2[2]=255;

	for(int i=0;i<im_flujo.size()-1;i++){		//elimino los ptos del flujo de status=0
		for(int j=0;j<ptos_encontrados[i].size();j++){
			if(status[i][j]==0){
				ptos_harris[i].erase(ptos_harris[i].begin()+j);
				ptos_encontrados[i].erase(ptos_encontrados[i].begin()+j);
				status[i].erase(status[i].begin()+j);
				j--;
			}
		}
	}
	for(int i=0;i<im_flujo.size()-1;i++){	
			CruzColor(1, im_flujo[i], ptos_harris[i], c2);
			CruzColor(1, im_flujo[i], ptos_encontrados[i], c2);
		for(int j=0;j<ptos_harris[i].size();j++){
			line(im_flujo[i], ptos_harris[i][j], ptos_encontrados[i][j], c2, 1, 8, 0);
		}
	}

	char *aux;
	char *tmp;
	for(int i=0;i<10;i++){
		aux=new char [100];
		tmp=new char [2];
		strcpy(aux,"Flujo Optico ");
		itoa(i, tmp, 10);
		strcat(aux, tmp);
		namedWindow(aux,1);
		imshow(aux, im_flujo[9-i]);
		waitKey();
		destroyWindow(aux);
	} 
	
}
int main ( int argc, char* argv[]){ 
	// insertar código 
	cout  << "OpenCV  detectada " <<  endl; 
 	Mat im= imread("imagenes/001.jpg",1); 
	vector<Point2f> c;
	c=APARTADO_A(im);
	c=APARTADO_B(im);
	APARTADO_C();
	APARTADO_D();
	

return 0; 
} 