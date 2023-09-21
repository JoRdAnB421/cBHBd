      program histo
      implicit none
      real r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r0,Dt,Dm
      real rho_DM,rho_GC,t,m,z,t0,incr1,incr2,Di1,Di2
      real rate(1000),a,b,d,zm,mtot_cl,f_dy,step,nm
      real x(1000),ym(1000),y10(1000),y90(1000),Nbin(200000,100)
      real xp1(1000),yp1(1000),Mcl(10000),rt,Mtot,mmax
      real xp2(1000),yp2(1000), CR, CG, CB,Dc,m0
      real nmass(1000000),mass(1000000),u1,u2,twopi
      integer i,l,h,ntot,n,CI,type,k,km,km0,k0
      real*8 bhout(20000000,20),harv,s,sigma!,nbin(200,1000)
      integer,dimension(10000):: INDEX
      twopi=3.14159265359*2.d0
      rho_DM=9.2d18
      rho_GC=rho_DM*5.d-5/1.e9      
      sigma=2.6e14/1.e9
      dt=5.
      
      open (unit=20,file='hmmodel-rapid-smallspins.dat')
      k=0
      do while(k.lt.98)
         k=k+1
         print*,k
      read(20,*,end=223,err=223)bhout(1,1:11),type,
     &     bhout(1,12:16)
      
      i=0
      mmax=0.d0
      do 
         i=i+1 
         read(20,*,end=223,err=223)bhout(i,1:11),type,
     &        bhout(i,12:16)
         if(bhout(i,13).ne.bhout(1,13))goto 222
c         if(bhout(i,3).gt.mmax)mmax=bhout(i,3)
      end do
 222  continue
      backspace(20)

      mmax=200.
      dm=4.
      m=3.d0-dm
      km=0
      do while(m.lt.mmax) 
         m0=m
         m=m+dm
         nm=0
         km=km+1
         do n=1,i
            if(bhout(n,3).gt.m0.and.bhout(n,3).le.m)then
               nm=nm+(bhout(n,12)/bhout(n,13))*bhout(n,14)
     &         *bhout(n,9)/bhout(n,11)*bhout(n,15)/bhout(n,16)/dm/dt!*rho_GC
            end if
         end do         
         write(21,*)m0,nm
         write(21,*)m,nm
         Nbin(km,k)=nm         
      end do
      end do
 223  continue      
      km0=km
      k0=k
      
      l=0
      km=0
      m=3.d0-dm
      do while(m.lt.mmax)
         km=km+1
         m0=m
         m=m+dm

         k=0
         do while(k.lt.k0)
            k=k+1           

!     generate more samples, including error on  DM density
            do  i=1,100
               call random(harv)
               u1 = harv
               call random(harv)
               u2 = harv
*     Generate two velocities from polar coordinates S & THETA.
               call random(harv)
               if(harv.gt.0.5)then
                  s=SQRT(-2.d0*LOG(u1))*cos(twopi*u2)*sigma
               else
                  s=-SQRT(-2.d0*LOG(u1))*cos(twopi*u2)*sigma
               end if
               rho_GC=7.3e14/1.e9+s
               nmass(i)=Nbin(km,k)*rho_GC
            end do
         end do         
         call SORTRX(i,nmass,INDEX)
         l=l+1
         x(l)=m0
         ym(l)=nmass(INDEX(int(i*0.5)))
         y10(l)=nmass(INDEX(int(i*0.05)))
         y90(l)=nmass(INDEX(int(i*0.95)))
         l=l+1
 5       x(l)=m
         ym(l)=nmass(INDEX(int(i*0.5)))
         y10(l)=nmass(INDEX(int(i*0.05)))
         y90(l)=nmass(INDEX(int(i*0.95)))                 
      end do

      call pgbegin (0,'plot-m.eps/vcps',0,0)
!     PLOT Gamma vs z
      call pgslw (2)         
      call pgvsize (1.,4.,1.,4.)     
      call pgwindow (3.,100.,-7.,1.5) !real(Rmax)*1.2)
      call pgbox ('bctSn',0.,0,'bctSn',0.,0)  
      call pglabel ('\fim\fn[\fiM\fn\d\(2281)\u]','log d\fiR\fn/d\fim\fn
     &[Gpc\u-3\dyr\u-1\d\fiM\fn\d\(2281)\u\u-1\d]','')    
      
      
      call pgsci(4)
      call pgslw(4)
      call pgsls(1)
      call pgline(l-1,x,log10(ym))
      call pgsls(1)
      call pgslw(1)
      call pgline(l-1,x,log10(y10))
      call pgline(l-1,x,log10(y90))

!     without HMs      
      open (unit=20,file='mmodel-rapid-smallspins.dat')
      k=0
      do while(k.lt.98)
         k=k+1 
         print*,k
      read(20,*,end=1223,err=1223)bhout(1,1:11),type,
     &     bhout(1,12:16)
      
      i=0
      mmax=0.d0
      do 
         i=i+1
         read(20,*,end=1223,err=1223)bhout(i,1:11),type,
     &        bhout(i,12:16)
         if(bhout(i,13).ne.bhout(1,13))goto 1222
c         if(bhout(i,3).gt.mmax)mmax=bhout(i,3)
      end do
 1222  continue
      backspace(20)

      mmax=200.
      dm=4.
      m=3.d0-dm
      km=0
      do while(m.lt.mmax) 
         m0=m
         m=m+dm
         nm=0
         km=km+1
         do n=1,i
            if(bhout(n,3).gt.m0.and.bhout(n,3).le.m)then
               nm=nm+(bhout(n,12)/bhout(n,13))*bhout(n,14)
     &           *bhout(n,9)/bhout(n,11)*bhout(n,15)/bhout(n,16)/dm/dt!*rho_GC
            end if
         end do         
         write(21,*)m0,nm
         write(21,*)m,nm
         Nbin(km,k)=nm         
      end do
      end do
 1223  continue      
      km0=km
      k0=k
      
      l=0
      km=0
      m=3.d0-dm
      do while(m.lt.mmax)
         km=km+1
         m0=m
         m=m+dm

         k=0
         do while(k.lt.k0)
            k=k+1           

!     generate more samples, including error on  DM density
            do  i=1,100
               call random(harv)
               u1 = harv
               call random(harv)
               u2 = harv
*     Generate two velocities from polar coordinates S & THETA.
               call random(harv)
               if(harv.gt.0.5)then
                  s=SQRT(-2.d0*LOG(u1))*cos(twopi*u2)*sigma
               else
                  s=-SQRT(-2.d0*LOG(u1))*cos(twopi*u2)*sigma
               end if
               rho_GC=7.3e14/1.e9+s
               nmass(i)=Nbin(km,k)*rho_GC
            end do
         end do         
         call SORTRX(i,nmass,INDEX)
         l=l+1
         x(l)=m0
         ym(l)=nmass(INDEX(int(i*0.5)))
         y10(l)=nmass(INDEX(int(i*0.05)))
         y90(l)=nmass(INDEX(int(i*0.95)))
         l=l+1
         x(l)=m
         ym(l)=nmass(INDEX(int(i*0.5)))
         y10(l)=nmass(INDEX(int(i*0.05)))
         y90(l)=nmass(INDEX(int(i*0.95)))                 
      end do

c      call pgbegin (0,'plot-m.eps/vcps',0,0)
!     PLOT Gamma vs z
c      call pgslw (2)         
c      call pgvsize (1.,4.,1.,4.)     
c      call pgwindow (3.,100.,-7.,.5) !real(Rmax)*1.2)
c      call pgbox ('bctSn',0.,0,'bctSn',0.,0)  
c      call pglabel ('\fim\fn[\fiM\fn\d\(2281)\u]','log d\fiR\fn/d\fim\fn
c     &[Gpc\u-3\dyr\u-1\d\fiM\fn\d\(2281)\u\u-1\d]','')    
      
      call pgsci(1)
      call pgslw(4)
      call pgsls(1)
      call pgline(l-1,x,log10(ym))
      call pgsls(1)
      call pgslw(1)
      call pgline(l-1,x,log10(y10))
      call pgline(l-1,x,log10(y90))
      
!     observed distribution      
      open(unit=143,file='bands2.txt') 
      i=0
      do
         i=i+1
         read(143,*,end=189)xp1(i),yp1(i)
      end do
 189  continue
      call pgscr(40,0.,0.7,0.5)
      call PGSFS(4)
      call pgsci(40) 
      call pgslw(1)
      call PGPOLY (i-1,xp1,log10(yp1))
      call pgsls(1)
      call pgline(i-1,xp1,log10(yp1))

      
      open(unit=143,file='media2.txt') 
      i=0
      do
         i=i+1
         read(143,*,end=129)xp1(i),yp1(i)
      end do
 129  continue
      call pgslw (4)
      call pgsls(1)
      call pgline(i-1,xp1,log10(yp1))
      call pgslw (2)
      call pgsci(1)
      

      call pgclos      
     
      end
      

      
      SUBROUTINE SORTRX(N,DATA,INDEX)
c     ordinamento veloce con quicksort  
c     piu' piccolo DATA(index(1))
c     piu' grande  DATA(index(N))
      INTEGER   N,INDEX(N)
      REAL      DATA(N)
       
      INTEGER   LSTK(31),RSTK(31),ISTK
      INTEGER   L,R,I,J,P,INDEXP,INDEXT
      REAL      DATAP
      
C     QuickSort Cutoff
C     

      INTEGER   M
      PARAMETER (M=9)
      
C===================================================================
C     
C     fai una stima iniziale dell'indice
      
      DO 50 I=1,N
         INDEX(I)=I
 50   CONTINUE
      
C     If array is short, skip QuickSort and go directly to
C     the straight insertion sort.
      
      IF (N.LE.M) GOTO 900
      
      ISTK=0
      L=1
      R=N
      
 200  CONTINUE
      
      
      I=L
      J=R
      
      
      P=(L+R)/2
      INDEXP=INDEX(P)
      DATAP=DATA(INDEXP)
      
      IF (DATA(INDEX(L)) .GT. DATAP) THEN
         INDEX(P)=INDEX(L)
         INDEX(L)=INDEXP
         INDEXP=INDEX(P)
         DATAP=DATA(INDEXP)
      ENDIF
      
      IF (DATAP .GT. DATA(INDEX(R))) THEN
         IF (DATA(INDEX(L)) .GT. DATA(INDEX(R))) THEN
            INDEX(P)=INDEX(L)
            INDEX(L)=INDEX(R)
         ELSE
            INDEX(P)=INDEX(R)
         ENDIF
         INDEX(R)=INDEXP
         INDEXP=INDEX(P)
         DATAP=DATA(INDEXP)
      ENDIF
      
      
 300  CONTINUE
      
      
      I=I+1
      IF (DATA(INDEX(I)).LT.DATAP) GOTO 300
      
 400  CONTINUE
      
      J=J-1
      IF (DATA(INDEX(J)).GT.DATAP) GOTO 400
      
C     Q5: collisione?
      
      IF (I.LT.J) THEN
         
C     Q6: interscambio DATA[I] <--> DATA[J] 
         
         INDEXT=INDEX(I)
         INDEX(I)=INDEX(J)
         INDEX(J)=INDEXT
         GOTO 300
      ELSE
         
C     Q7: Yes, select next subsequence to sort
C     
C     A questo punto, I >= J ae DATA[l] <= DATA[I] == DATAP <= DATA[r],
C     per tutti L <= l < I e J < r <= R.  
         
         IF (R-J .GE. I-L .AND. I-L .GT. M) THEN
            ISTK=ISTK+1
            LSTK(ISTK)=J+1
            RSTK(ISTK)=R
            R=I-1
         ELSE IF (I-L .GT. R-J .AND. R-J .GT. M) THEN
            ISTK=ISTK+1
            LSTK(ISTK)=L
            RSTK(ISTK)=I-1
            L=J+1
         ELSE IF (R-J .GT. M) THEN
            L=J+1
         ELSE IF (I-L .GT. M) THEN
            R=I-1
         ELSE
C     Q8: Pop the stack, or terminate QuickSort if empty
            IF (ISTK.LT.1) GOTO 900
            L=LSTK(ISTK)
            R=RSTK(ISTK)
            ISTK=ISTK-1
         ENDIF
         GOTO 200
      ENDIF
      
 900  CONTINUE
      
C===================================================================
C     
C     Q9: Straight Insertion sort
      
      DO 950 I=2,N
         IF (DATA(INDEX(I-1)) .GT. DATA(INDEX(I))) THEN
            INDEXP=INDEX(I)
            DATAP=DATA(INDEXP)
            P=I-1
 920        CONTINUE
            INDEX(P+1) = INDEX(P)
            P=P-1
            IF (P.GT.0) THEN
               IF (DATA(INDEX(P)).GT.DATAP) GOTO 920
            ENDIF
            INDEX(P+1) = INDEXP
         ENDIF
 950  CONTINUE
      
C===================================================================
C     
C     All done
      
      END


      SUBROUTINE random(rnd)
!     genera un numero casuale tra 0 ed 1
      IMPLICIT NONE
      REAL(8),INTENT(OUT) :: rnd
      INTEGER :: num = 10
      INTEGER :: isize,idate(8)
      INTEGER,ALLOCATABLE :: iseed(:)
      INTEGER :: i
      CALL DATE_AND_TIME(VALUES=idate)
      CALL RANDOM_SEED(SIZE=isize) !imposta il numero di interi per contenere il seme      
      ALLOCATE( iseed(isize) )
      CALL RANDOM_SEED(GET=iseed) !acquisisce il valore corrente del seme
      iseed = iseed * (idate(8)-1000) ! idate(8) contains milisecond
      CALL RANDOM_SEED(PUT=iseed) !imposta il nuovo seme            
      CALL RANDOM_NUMBER(rnd)
      DEALLOCATE(iseed)           
      END SUBROUTINE random
