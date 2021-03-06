int w = 600;
int h = 600;
float b1, b2;
float vb1, vb2;
float x1, x2, x3, x4;
float y1, y2, y3, y4;
float vx1, vx2, vx3, vx4;
float vy1, vy2, vy3, vy4;
float vb = 1;
float v = 0.5;
float d1 = 50;
float d2 = 75;
float dl = 15;
float lx;
float ly;
float d = (d1+d2)/2 - 5;
float ld = (d1+dl)/2 - 1;
int a = 0;
int lf;
boolean lp;
boolean la;
boolean pause;
int m, m1, ml, ms;
PImage img, str, end, restr;
PFont font, f;
String s1 = "Click mouse to start the game !!!";
String s2 = "Game Over!!! Click mouse to restart.";

void reset() {
  vb1 = vb2 = vx1 = vx2 = vx3 = vx4 = vy1 = vy2 = vy3 = vy4 = 0;
  b1 = w/2;
  b2 = h/2;
  x1 = w/8;
  y1 = h/6;
  x2 = 7*w/8;
  y2 = h/6;
  x3 = w/8;
  y3 = 7*h/8;
  x4 = 7*w/8;
  y4 = 7*h/8;
  m = millis();
  ms = m;
  ml = m;
  lf = 3;
  lp = false;
  la = false;
  pause = false;
}


void setup()
{
  size(600, 600);
  smooth();
  img = loadImage("game2.jpg");
  str = loadImage("start.png");
  end = loadImage("end.png");
  restr = loadImage("restart.png");
  font = createFont("Serif", height/15);
  textFont(font);
}

void draw()
{
  image(img, 0, 0, width, height);
  switch(a) {
  case 0:
    Screen_0();
    break;
  case 1:
    Screen_1();
    break;
  case 2:
    Screen_2();
  }
}

void Screen_0() {
  image(str, width/2-100, height/2-50, 200, 100);
}

void Screen_1() {
  m = millis();
  fill(210, 187, 187);
  rect(0, 0, width, height/20);

  f = createFont("Serif", 20);
  textFont(f);
  fill(0);
  int hr = (m-ms)/3600000;
  int min = ((m-ms)%3600000)/60000;
  int sec = (((m-ms)%3600000)%60000)/1000;
  text(hr+":"+min+":"+sec, width-100, 2, 80, 50);
  text("Life: "+lf, width/2, 2, 80, 50);

  fill(127, 255, 0);
  ellipse(b1, b2, d1, d1);
  fill(127, 0, 255);
  ellipse(x1, y1, d2, d2);
  ellipse(x2, y2, d2, d2);
  ellipse(x3, y3, d2, d2);
  ellipse(x4, y4, d2, d2);

  if (sqrt(sq(x1-b1) + sq(y1-b2)) < d || sqrt(sq(x2-b1) + sq(y2-b2)) < d || sqrt(sq(x3-b1) + sq(y3-b2)) < d || sqrt(sq(x4-b1) + sq(y4-b2)) < d || b1<(0.5*d1-5) || b1>(width-0.5*d1+5)  || b2<(0.5*d1+25) || b2>(height-0.5*d1+5)) 
    a = 2;

  if (lf != 0 && lp == false) {
    if (la==false && (m-ml)/1000 > 20) {
      lx = random(dl, width-dl);
      ly = random(dl+height/20, height-dl);
      la = true;
    }
    fill(255, 0, 127);
    ellipse(lx, ly, dl, dl);
  }

  if (sqrt(sq(lx-b1) + sq(ly-b2)) < ld) {
    ml = m;
    lf--;
    lp = true;
    la = false;
    lx = ly = 0;
    vx1 = vx2 = vx3 = vx4 = vy1 = vy2 = vy3 = vy4 = 0;
  } else if (lp==true && (m-ml)/1000 < 5);
  else if (pause==false) {
    lp = false;
    if (x1<b1) {
      vx1 = v/(sqrt(1+sq(((y1-b2)/(x1-b1)))));
    } else {
      vx1 = -v/(sqrt(1+sq(((y1-b2)/(x1-b1)))));
    }

    if (x2<b1) {
      vx2 = v/(sqrt(1+sq(((y2-b2)/(x2-b1)))));
    } else {
      vx2 = -v/(sqrt(1+sq(((y2-b2)/(x2-b1)))));
    }

    if (x3<b1) {
      vx3 = v/(sqrt(1+sq(((y3-b2)/(x3-b1)))));
    } else {
      vx3 = -v/(sqrt(1+sq(((y3-b2)/(x3-b1)))));
    }

    if (x4<b1) {
      vx4 = v/(sqrt(1+sq(((y4-b2)/(x4-b1)))));
    } else {
      vx4 = -v/(sqrt(1+sq(((y4-b2)/(x4-b1)))));
    }

    vy1 = ((y1-b2)/(x1-b1))*vx1;
    vy2 = ((y2-b2)/(x2-b1))*vx2;
    vy3 = ((y3-b2)/(x3-b1))*vx3;
    vy4 = ((y4-b2)/(x4-b1))*vx4;
  }

  x1 = x1 + vx1;
  y1 = y1 + vy1;
  x2 = x2 + vx2;
  y2 = y2 + vy2;
  x3 = x3 + vx3;
  y3 = y3 + vy3;
  x4 = x4 + vx4;
  y4 = y4 + vy4;
  b1 = b1 + vb1;
  b2 = b2 + vb2;
}

void Screen_2() {
  image(end, width/2-100, height/4-50, 200, 100);
  image(restr, width/2-100, height/4+100, 200, 100);
}

void keyPressed() {
  if (key == CODED) {
    if (keyCode == UP) {
      vb2 = -vb;
      vb1 = 0;
    } else if (keyCode == DOWN) {
      vb2 = vb;
      vb1 = 0;
    } else if (keyCode == LEFT) {
      vb1 = -vb;
      vb2 = 0;
    } else if (keyCode == RIGHT) {
      vb1 = vb;
      vb2 = 0;
    }
  } else if (key == ' ' && pause==false) {
    vb1 = vb2 = vx1 = vx2 = vx3 = vx4 = vy1 = vy2 = vy3 = vy4 = 0;
    pause=true;
  } else if (key == ' ') {
    pause=false;
  }
}

void mouseClicked()
{
  if (a==0 && mouseX > width/2-100 && mouseX < width/2+100 && mouseY > height/2-50 && mouseY < height/2+50) {
    a = 1;
    reset();
  } else if (a==2 && mouseX > width/2-100 && mouseX < width/2+100 && mouseY > height/4+100 && mouseY < height/4+200) {
    a = 1;
    reset();
  }
  draw();
}

